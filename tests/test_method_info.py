import unittest
from urllib.parse import urlparse

from chrono_stream.literature_reviews import METHOD_LITERATURE_REVIEWS
from chrono_stream.method_info import METHOD_INFORMATION, copy_ready_method_note
from chrono_stream.registry import MODEL_NAMES


class MethodInformationTests(unittest.TestCase):
    def test_every_model_has_complete_multi_source_information(self) -> None:
        self.assertEqual(set(METHOD_INFORMATION), set(MODEL_NAMES))
        self.assertEqual(set(METHOD_LITERATURE_REVIEWS), set(MODEL_NAMES))
        for model_id, title in MODEL_NAMES.items():
            with self.subTest(model=model_id):
                information = METHOD_INFORMATION[model_id]
                self.assertGreaterEqual(len(information.references), 2)
                self.assertGreater(len(information.origin), 120)
                self.assertGreater(len(information.how_it_works), 120)
                self.assertGreater(len(information.chrono_stream), 120)
                self.assertGreater(len(information.limitations), 120)
                self.assertIn("(", information.citation_ready)
                literature_review = METHOD_LITERATURE_REVIEWS[model_id]
                self.assertGreater(len(literature_review), 1_500)
                self.assertGreaterEqual(literature_review.count("\n\n"), 2)

                source_urls = [reference.url for reference in information.references]
                self.assertEqual(len(source_urls), len(set(source_urls)))
                for reference in information.references:
                    parsed = urlparse(reference.url)
                    self.assertEqual(parsed.scheme, "https")
                    self.assertTrue(parsed.netloc)
                    self.assertGreater(len(reference.apa), 50)
                    self.assertGreater(len(reference.contribution), 40)
                    self.assertNotIn("wikipedia.org", parsed.netloc.lower())
                    self.assertNotIn("researchgate.net", parsed.netloc.lower())

                note = copy_ready_method_note(model_id, title)
                self.assertIn(title, note)
                self.assertIn("Overview", note)
                self.assertIn("Literature review", note)
                self.assertIn(literature_review, note)
                self.assertIn("References (APA 7)", note)
                self.assertNotIn("\n\n\n", note)
                self.assertNotIn("Source-to-method audit", note)
                for reference in information.references:
                    self.assertIn(reference.apa, note)

    def test_arima_and_sarima_document_the_complete_component_chain(self) -> None:
        arima = METHOD_INFORMATION["arima"]
        sarima = METHOD_INFORMATION["sarima"]
        self.assertGreaterEqual(len(arima.references), 20)
        self.assertGreater(len(sarima.references), len(arima.references))
        for term in (
            "Box–Cox",
            "Yeo–Johnson",
            "ADF",
            "KPSS",
            "Phillips–Perron",
            "ACF/PACF",
            "AICc",
            "Ljung",
            "ARCH",
            "zero residual mean",
        ):
            self.assertIn(term, arima.chrono_stream)
        for term in ("OCSB", "Canova–Hansen", "seasonal differencing"):
            self.assertIn(term, sarima.chrono_stream)

    def test_stl_note_does_not_misrepresent_x11(self) -> None:
        information = METHOD_INFORMATION["stl"]
        self.assertIn("not official X-11", information.limitations)
        self.assertIn("not an implementation of Census X-11", information.citation_ready)
        self.assertIn("actually executed", information.references[-1].contribution)

    def test_new_method_notes_preserve_source_and_implementation_boundaries(self) -> None:
        mstl = METHOD_INFORMATION["mstl_ets"]
        self.assertIn("not a forecasting algorithm prescribed", mstl.limitations)
        self.assertIn("decomposition rather than a complete forecast rule", mstl.citation_ready)

        nbeats = METHOD_INFORMATION["nbeats"]
        self.assertIn("generic N-BEATS", nbeats.chrono_stream)
        self.assertIn("does not reproduce", nbeats.citation_ready)

        tcn = METHOD_INFORMATION["tcn"]
        self.assertIn("not WaveNet", tcn.limitations)
        self.assertIn("rather than reproducing", tcn.citation_ready)

        croston = METHOD_INFORMATION["croston_family"]
        self.assertIn("Exact zero means no demand", croston.chrono_stream)
        self.assertEqual(len(croston.references), 3)


if __name__ == "__main__":
    unittest.main()
