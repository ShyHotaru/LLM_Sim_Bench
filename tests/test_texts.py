import contextlib
import io
import json
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

from llm_sim_bench import export, texts
from llm_sim_bench.cli import main
from llm_sim_bench.mock import MockEngine
from llm_sim_bench.server import make_handler, reply_for


class ChunkTests(unittest.TestCase):
    def test_js_round_rounds_halves_up(self):
        self.assertEqual([texts.js_round(x) for x in (0.5, 1.5, 2.5, 2.4999, 0.49999999999999994)],
                         [1, 2, 3, 2, 0])

    def test_pieces_keep_the_text_and_match_the_ratio(self):
        en = "Alice was beginning to get very tired of sitting by her sister on the bank."
        pieces = texts.chunk_text(en, 4)
        self.assertEqual("".join(pieces), en)
        self.assertEqual(len(pieces), texts.js_round(len(en) / 4))
        self.assertGreaterEqual(sum(p.startswith(" ") for p in pieces), len(en.split()) - 2)
        ko = "가을 하늘 공활한데"
        bytey = texts.chunk_text(ko, 0.7)
        self.assertEqual("".join(bytey), ko)
        self.assertIn("", bytey)
        self.assertEqual(texts.chunk_text("", 3), [])
        self.assertEqual(texts.chunk_text("\U0001F600" * 3, 1), ["\U0001F600"] * 3)

    def test_unknown_tokenizer_uses_the_median(self):
        table = {"chars_per_token": {"t": {"a": 1, "b": 2, "c": 4}, "u": {"a": 1, "b": 3}}}
        self.assertEqual(texts.chars_per_token("t", "b", table), (2, True))
        self.assertEqual(texts.chars_per_token("t", "", table), (2, False))
        self.assertEqual(texts.chars_per_token("u", "", table), (2, False))
        self.assertEqual(texts.chars_per_token("none", "a", table), (4, False))

    def test_guess_lang(self):
        self.assertEqual(texts.guess_lang("RTX 4090의 VRAM은 24GB입니다"), "ko")
        self.assertEqual(texts.guess_lang("吾輩は猫である"), "ja")
        self.assertEqual(texts.guess_lang("Alice was tired."), "en")
        self.assertIsNone(texts.guess_lang("12345 !?"))


class TextDataTests(unittest.TestCase):
    def test_package_copies_and_web_golden_are_current(self):
        self.assertEqual(export.stale(), [])

    def test_every_text_loads_with_a_ratio_for_every_model(self):
        from llm_sim_bench.profiles import MODEL_PRESETS
        families = {m.tokenizer for m in MODEL_PRESETS.values()}
        for e in texts.manifest()["texts"]:
            for family in families:
                s = texts.script(family, e["id"])
                self.assertTrue(s.measured, f"{e['id']} / {family}")
                self.assertTrue(all(a for _, a in s.turns))

    def test_public_domain_texts_have_attribution(self):
        for e in texts.manifest()["texts"]:
            line = texts.credit_line(e)
            if e["origin"] == "public-domain":
                self.assertIn(e["source"]["page"], line)
                self.assertIn(e["author"], texts.describe(e))
        # Aozora Bunko asks that the edition and transcription notes be kept.
        neko = texts.describe(texts.entry("ja-neko"))
        self.assertIn("底本", neko)
        self.assertIn("入力", neko)
        self.assertIn("校正", neko)

    def test_chat_script_and_text_file(self):
        s = texts.script("llama3", "en-chat")
        turns = json.loads((export.TEXTS_DIR / "en-chat.json").read_text("utf-8"))["turns"]
        self.assertEqual(len(s.turns), len(turns))
        self.assertEqual("".join(s.turns[1][0]), turns[1]["user"])
        self.assertEqual("".join(s.turns[1][1]), turns[1]["assistant"])
        with self.assertRaises(ValueError):
            texts.script("llama3", "no-such-text")
        f = export.TEXTS_DIR / "ko-answer.txt"
        custom = texts.script("llama3", path=str(f))
        self.assertEqual(custom.ratio_id, "ko-answer")
        self.assertEqual(custom.entry["origin"], "custom")


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.script = texts.script("llama3", "en-chat")
        engine = MockEngine(prefill_tps=1e6, decode_tps=lambda _: 1e5, jitter=0)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0),
                                         make_handler(engine, "m", 1000, self.script))
        self.httpd.RequestHandlerClass.log_message = lambda *a: None
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1/chat/completions"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, body):
        req = urllib.request.Request(self.url, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.read().decode()

    def test_chat_follows_the_script_by_user_turn(self):
        msgs = [{"role": "system", "content": "x"}, {"role": "user", "content": "hi"}]
        out = json.loads(self.post({"messages": msgs}))
        first = "".join(self.script.turns[0][1])
        self.assertEqual(out["choices"][0]["message"]["content"], first)
        self.assertEqual(out["choices"][0]["finish_reason"], "stop")
        self.assertEqual(out["usage"]["completion_tokens"], len(self.script.turns[0][1]))
        msgs += [{"role": "assistant", "content": first}, {"role": "user", "content": "more"}]
        out = json.loads(self.post({"messages": msgs, "max_tokens": 5}))
        self.assertEqual(out["choices"][0]["message"]["content"],
                         "".join(self.script.turns[1][1][:5]))
        self.assertEqual(out["choices"][0]["finish_reason"], "length")
        many = [{"role": "user", "content": "x"}] * (len(self.script.turns) + 1)
        self.assertIs(reply_for(self.script, many), self.script.turns[0][1])

    def test_stream(self):
        raw = self.post({"messages": [{"role": "user", "content": "hi"}], "stream": True})
        events = [json.loads(line[6:]) for line in raw.splitlines()
                  if line.startswith("data: {")]
        self.assertTrue(raw.rstrip().endswith("data: [DONE]"))
        self.assertEqual(events[0]["choices"][0]["delta"]["role"], "assistant")
        text = "".join(e["choices"][0]["delta"].get("content", "") for e in events)
        self.assertEqual(text, "".join(self.script.turns[0][1]))
        self.assertEqual(events[-1]["choices"][0]["finish_reason"], "stop")


class CliTextTests(unittest.TestCase):
    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_run_streams_a_chat(self):
        code, out, err = self.run_cli("run", "--text", "ja-chat", "--decode-tps", "100000",
                                      "--prefill-tps", "1000000", "--max-tokens", "3")
        self.assertEqual(code, 0)
        self.assertIn("User: ", out)
        self.assertIn("turn 4/4", out)
        self.assertIn("written for LLM Sim Bench", err)

    def test_texts_and_unknown_text(self):
        code, out, _ = self.run_cli("texts", "en-alice")
        self.assertEqual(code, 0)
        self.assertIn("https://www.gutenberg.org/ebooks/11", out)
        code, _, err = self.run_cli("run", "--text", "nope", "--decode-tps", "1e5")
        self.assertEqual(code, 1)
        self.assertIn("unknown text", err)


if __name__ == "__main__":
    unittest.main()
