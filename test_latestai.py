import gzip
import json
import re
import tempfile
import unittest
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import latestai

NOW = datetime(2026, 11, 5, 12, tzinfo=timezone.utc)
CONFIG = json.loads(latestai.CONFIG_PATH.read_text())
RULES = CONFIG["huggingface"]
WATCHED = {"black-forest-labs"}


def model(id_="someone/model", days_old=1, **extra):
    created = (NOW - timedelta(days=days_old)).isoformat()
    return {"id": id_, "pipeline_tag": "text-to-video", "createdAt": created, "tags": [], **extra}


class KeepModelTests(unittest.TestCase):
    def keep(self, m):
        return latestai.keep_model(m, RULES, WATCHED, NOW)

    def test_new_unknown_model_without_traction_is_dropped(self):
        self.assertFalse(self.keep(model(likes=2)))

    def test_new_unknown_model_with_traction_is_kept(self):
        self.assertTrue(self.keep(model(likes=60)))

    def test_watched_org_original_is_kept_without_traction(self):
        self.assertTrue(self.keep(model("black-forest-labs/new")))

    def test_watched_org_lora_needs_some_traction(self):
        lora = model("black-forest-labs/style", tags=["lora"])
        self.assertFalse(self.keep(lora))
        self.assertTrue(self.keep({**lora, "trendingScore": 10}))

    def test_community_derivative_needs_more_traction_than_original(self):
        tags = ["base_model:quantized:a/b"]
        self.assertFalse(self.keep(model(tags=tags, likes=60)))
        self.assertTrue(self.keep(model(tags=tags, likes=200)))

    def test_old_model_is_dropped_however_popular(self):
        self.assertFalse(self.keep(model("black-forest-labs/flux", days_old=400, likes=9000)))

    def test_nsfw_is_dropped_by_tag_and_by_name(self):
        self.assertFalse(self.keep(model(likes=999, tags=["not-for-all-audiences"])))
        self.assertFalse(self.keep(model("a/remix-NSFW-v2", likes=999)))
        self.assertFalse(self.keep(model("a/qwen-uncensored", likes=999)))

    def test_other_pipelines_are_dropped(self):
        self.assertFalse(self.keep({**model(likes=999), "pipeline_tag": "text-generation"}))


class MergeTests(unittest.TestCase):
    def test_metrics_from_different_sorts_are_combined(self):
        merged = latestai.merge_models([
            {"id": "a/b", "likes": 5},
            {"id": "a/b", "likes": 5, "trendingScore": 30},
        ])
        self.assertEqual(merged, [{"id": "a/b", "likes": 5, "trendingScore": 30}])


class FeedTests(unittest.TestCase):
    RSS = b"""<rss><channel><title>Feed title is not an item</title><item>
      <title>New &lt;b&gt;Veo&lt;/b&gt; model</title><link>https://x.test/1</link>
      <pubDate>Wed, 04 Nov 2026 11:00:00 GMT</pubDate><description>&lt;p&gt;Hi&lt;/p&gt;</description>
    </item></channel></rss>"""
    ATOM = b"""<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>v1.0</title>
      <link rel="self" href="https://x.test/self"/><link rel="alternate" href="https://x.test/2"/>
      <updated>2026-11-04T01:12:27Z</updated><content>notes</content></entry></feed>"""

    def test_rss(self):
        self.assertEqual(latestai.parse_feed(self.RSS), [{
            "title": "New Veo model", "url": "https://x.test/1",
            "date": "Wed, 04 Nov 2026 11:00:00 GMT", "summary": "Hi",
        }])

    def test_atom_uses_alternate_link(self):
        entry = latestai.parse_feed(self.ATOM)[0]
        self.assertEqual(entry["url"], "https://x.test/2")
        self.assertEqual(entry["summary"], "notes")

    def select(self, **entry):
        base = {"title": "Veo ships", "url": "https://x.test/1",
                "date": "2026-11-04T00:00:00Z", "summary": ""}
        feed = {"name": "F", "filter": "media"}
        return latestai.select_feed_entries([{**base, **entry}], feed, CONFIG, NOW)

    def test_filter_and_age(self):
        self.assertEqual(len(self.select()), 1)
        self.assertEqual(self.select(title="Quarterly earnings"), [])
        self.assertEqual(self.select(date="2026-01-01T00:00:00Z"), [])

    def test_financial_runway_is_not_a_match(self):
        self.assertEqual(self.select(title="Startup extends its runway"), [])

    def test_title_prefix_labels_bare_versions_without_doubling(self):
        feed = {"name": "F", "title_prefix": "Diffusers "}
        base = {"url": "https://x.test/1", "date": "2026-11-04T00:00:00Z", "summary": ""}
        titles = [
            latestai.select_feed_entries([{**base, "title": t}], feed, CONFIG, NOW)[0]["title"]
            for t in ("v1.0", "Diffusers 1.0")
        ]
        self.assertEqual(titles, ["Diffusers v1.0", "Diffusers 1.0"])

    def test_reused_links_still_give_each_update_its_own_id(self):
        base = {"url": "https://x.test/video", "date": "2026-11-04T00:00:00Z", "summary": ""}
        entries = [{**base, "title": "Video: Model A"}, {**base, "title": "Video: Model B"}]
        reused = {"name": "F", "links_reused": True}
        ids = {i["id"] for i in latestai.select_feed_entries(entries, reused, CONFIG, NOW)}
        self.assertEqual(len(ids), 2)
        # Other feeds keep link-only IDs, so items already seen don't come back as new.
        plain = latestai.select_feed_entries(entries[:1], {"name": "F"}, CONFIG, NOW)
        self.assertEqual(plain[0]["id"], "feed:https://x.test/video")

    def test_unsafe_link_scheme_is_rejected(self):
        self.assertEqual(self.select(url="javascript:alert(1)"), [])


class MediaFilterTests(unittest.TestCase):
    def matches(self, text):
        return bool(re.search(CONFIG["filters"]["media"], text))

    def test_matches_releases_and_editing_tools(self):
        for text in (
            "OpenAI Launches GPT-Image-2.6 Flare",
            "ChatGPT can now make animated GIFs with Images 2.6",
            "MoviEdit: Long-form Multi-Shot Video Editing",
            "Premiere adds new AI masking",
            "Google brings Veo to Vids",
        ):
            self.assertTrue(self.matches(text), text)

    def test_ignores_general_ai_and_camera_news(self):
        for text in (
            "Baidu releases ERNIE 5.5 Flash",
            "Nikon ZR Firmware 2.00 Detailed",
            "Agents Make Inference Up to 2.6x Faster",
        ):
            self.assertFalse(self.matches(text), text)


class FetchTests(unittest.TestCase):
    def test_gzip_body_is_decompressed_without_a_content_encoding_request(self):
        body = gzip.compress(b"<rss/>")
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = body
        with mock.patch("urllib.request.urlopen", return_value=response):
            self.assertEqual(latestai.fetch("https://x.test/"), b"<rss/>")


class PaperTests(unittest.TestCase):
    def papers(self, **paper):
        entry = {"summary": "We also generate video generation samples.", "paper": {
            "id": "1", "title": "Robot routing", "upvotes": 50, "ai_keywords": [], **paper}}
        with mock.patch.object(latestai, "fetch_json", return_value=[entry]):
            return latestai.fetch_hf_papers(CONFIG)

    def test_passing_mention_in_abstract_does_not_match(self):
        self.assertEqual(self.papers(), [])

    def test_keyword_matches(self):
        self.assertEqual(len(self.papers(ai_keywords=["text-to-image"])), 1)

    def test_low_upvotes_are_dropped(self):
        self.assertEqual(self.papers(title="Video generation", upvotes=1), [])


class TryOptionTests(unittest.TestCase):
    RULES = CONFIG["try"]

    def space(self, id_="a/demo", stage="RUNNING", hardware="zero-a10g", **extra):
        return {"id": id_, "runtime": {"stage": stage, "hardware": {"current": hardware}}, **extra}

    def test_ranking_prefers_owner_and_drops_unrelated_spaces(self):
        ranked = latestai.rank_spaces(
            "bfl/flux-klein",
            ["x/vram-calculator", "y/ImageGen-32", "z/flux-klein-fast", "bfl/playground", "t/flux-trainer"],
            self.RULES,
        )
        self.assertEqual(ranked, ["bfl/playground", "z/flux-klein-fast"])

    def test_generic_words_and_versions_do_not_link_a_space(self):
        self.assertEqual(latestai.rank_spaces("a/Foo-Image-2.0", ["b/image-2-0-tool"], self.RULES), [])

    def test_zerogpu_space_needs_no_signup(self):
        demo = latestai.describe_space(self.space(), "a/model")
        self.assertIn("No signup", demo["access"])
        self.assertTrue(demo["official"])

    def test_oauth_space_never_claims_no_signup(self):
        demo = latestai.describe_space(self.space(cardData={"hf_oauth": True}), "b/model")
        self.assertIn("login", demo["access"])
        self.assertNotIn("No signup", demo["access"])
        self.assertFalse(demo["official"])

    def test_broken_or_restricted_spaces_are_not_offered(self):
        for space in (
            self.space(stage="RUNTIME_ERROR"),
            self.space(stage="BUILD_ERROR"),
            self.space(stage="PAUSED"),
            self.space(disabled=True),
            self.space(private=True),
            self.space(gated="auto"),
        ):
            self.assertIsNone(latestai.describe_space(space, "a/model"), space)

    def test_sleeping_space_is_offered_with_a_warning(self):
        demo = latestai.describe_space(self.space(stage="SLEEPING"), "a/model")
        self.assertIn("asleep", demo["access"])

    def test_only_live_providers_are_listed(self):
        info = {"inferenceProviderMapping": {
            "fal": {"status": "live"}, "old": {"status": "staging"}}}
        self.assertEqual(latestai.live_providers(info), ["fal"])

    def test_failed_lookup_keeps_the_stored_stats(self):
        item = {"id": "hf:a/b", "title": "a/b", "trending": 7, "likes": 3}
        with mock.patch.object(latestai, "fetch_json", side_effect=OSError("down")):
            errors = latestai.add_try_options([item], CONFIG)
        self.assertEqual((item["trending"], item["likes"]), (7, 3))
        self.assertIn("Try links for a/b", errors[0])

    def test_live_stats_replace_stored_stats(self):
        item = {"id": "hf:a/b", "title": "a/b", "trending": 7, "likes": 3}
        info = {"trendingScore": 90, "likes": 44}
        with mock.patch.object(latestai, "fetch_json", return_value=info):
            self.assertEqual(latestai.add_try_options([item], CONFIG), [])
        self.assertEqual((item["trending"], item["likes"]), (90, 44))

    def test_deleted_space_is_skipped(self):
        info = {"spaces": ["a/gone", "a/live"]}

        def fetch_space(space):
            if space == "a/gone":
                raise urllib.error.HTTPError(space, 404, "gone", None, None)
            return self.space(space)

        with mock.patch.object(latestai, "fetch_json", return_value=info):
            options = latestai.find_try_options({"title": "a/model"}, self.RULES, fetch_space)
        self.assertEqual([d["id"] for d in options["demos"]], ["a/live"])

    def test_render_covers_paid_gated_and_no_demo_and_escapes(self):
        page = latestai.render_try(
            {"demos": [], "providers": ["<evil>"], "gated": "auto"}, "https://huggingface.co/a/b"
        )
        self.assertIn("No live demo", page)
        self.assertIn("paid credits", page)
        self.assertIn("accepting the terms", page)
        self.assertNotIn("<evil>", page)

    def test_render_links_demo(self):
        demo = {"id": "a/b", "url": "https://huggingface.co/spaces/a/b", "access": "No signup needed", "official": True}
        page = latestai.render_try({"demos": [demo], "providers": [], "gated": False}, "https://x.test")
        self.assertIn("Try official demo", page)
        self.assertIn('href="https://huggingface.co/spaces/a/b"', page)
        self.assertIn("a/b: No signup needed", page)


class StateTests(unittest.TestCase):
    def test_item_is_new_once_and_keeps_first_seen_with_updated_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = latestai.open_db(Path(tmp) / "state.db")
            item = {"id": "x", "meta": ["1 likes"]}
            self.assertEqual(latestai.record_items(db, [item], NOW), [item])
            later = NOW + timedelta(days=1)
            self.assertEqual(latestai.record_items(db, [{"id": "x", "meta": ["9 likes"]}], later), [])
            stored = latestai.recent_items(db, 14, later)[0]
            self.assertEqual(stored["meta"], ["9 likes"])
            self.assertEqual(stored["first_seen"], NOW.isoformat())
            db.close()


class RenderTests(unittest.TestCase):
    def item(self, id_, section="Video generation", date="2026-11-01T00:00:00+00:00", **extra):
        return {"id": id_, "section": section, "title": id_, "url": "https://x.test/", "source": "S",
                "date": date, "meta": [], "summary": "", **extra}

    def order(self, items, new_ids=frozenset()):
        page = latestai.render_digest(items, set(new_ids), [], 14, NOW)
        return sorted((i["id"] for i in items), key=lambda name: page.index(f">{name}</a>"))

    def test_models_rank_by_trending_even_if_older_or_not_new(self):
        items = [
            self.item("low", trending=5, likes=900, date="2026-11-04T00:00:00+00:00"),
            self.item("high", trending=300, likes=1, date="2026-10-10T00:00:00+00:00"),
            self.item("mid", trending=40.5, likes=2),
        ]
        self.assertEqual(self.order(items, {"low"}), ["high", "mid", "low"])

    def test_likes_break_trending_ties(self):
        items = [self.item("few", trending=9, likes=1), self.item("many", trending=9, likes=8)]
        self.assertEqual(self.order(items), ["many", "few"])

    def test_news_stays_newest_first(self):
        items = [
            self.item("older", "News and tool updates", "2026-11-01T00:00:00+00:00"),
            self.item("newer", "News and tool updates", "2026-11-04T00:00:00+00:00"),
        ]
        self.assertEqual(self.order(items), ["newer", "older"])

    def test_stats_are_shown_including_zero_trending(self):
        page = latestai.render_digest([self.item("m", trending=0, likes=1)], set(), [], 14, NOW)
        self.assertIn("trending 0 · 1 like ·", page)

    def test_feed_content_is_escaped(self):
        item = {"id": "i", "section": "Research", "title": "<script>x</script>",
                "url": 'https://x.test/"onmouseover="y', "source": "S", "date": None,
                "meta": [], "summary": "<img src=x>"}
        page = latestai.render_digest([item], {"i"}, ["bad <source>"], 14, NOW)
        self.assertNotIn("<script>x", page)
        self.assertNotIn("<img", page)
        self.assertNotIn('"onmouseover', page)
        self.assertNotIn("<source>", page)


if __name__ == "__main__":
    unittest.main()
