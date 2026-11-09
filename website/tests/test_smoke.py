"""Small real-model smoke test; run with `python -m unittest discover -s tests`."""
import asyncio
import math
import unittest
from io import BytesIO
from pathlib import Path

import pandas as pd
from pydantic import ValidationError
from starlette.requests import Request

import app as web


def example_rows():
    ages = (2000, 4000, 7000, 10000)
    rows = []
    lam230 = 9.17055e-6
    lam234 = 2.82203e-6
    for i, age in enumerate(ages):
        u48, th28 = 1.08, 0.003 + 0.0005 * i
        th08 = (1 - math.exp(-lam230 * age) * (1 - 0.83 * th28)
                + (u48 - 1) * lam230 / (lam230 - lam234)
                * (1 - math.exp(-(lam230 - lam234) * age)))
        rows.append(dict(Depth=10 + i * 10, U234_r=u48, U234_r_err=0.0008,
                         Th230_r=th08, Th230_r_err=0.0005,
                         Th232_r=th28, Th232_r_err=0.0001, Depth_err=0.1))
    return rows


class WebSmokeTest(unittest.TestCase):
    def test_fraction_mapping(self):
        settings = web.Settings(fraction_det=0.7)
        self.assertAlmostEqual(web._meta(settings)["fraction_bulk"], 0.3)

    def test_faraji_reference_and_excel_style_headers(self):
        example = Path(web.__file__).parent / "static/examples/faraji_2021.csv"
        frame = pd.read_csv(example)
        self.assertEqual(len(frame), 10)
        self.assertEqual(frame["Faraji_initial"].tolist(),
                         [8.05, 6.00, 5.98, 6.63, 7.74, 6.55, 6.01, 6.21, 6.87, 4.88])
        self.assertEqual(frame["Faraji_err"].tolist(),
                         [0.24, 0.25, 0.27, 0.25, 0.18, 0.21, 0.50, 0.46, 0.27, 0.97])
        # The actual paper workbook uses underscore-separated ratio headings.
        alternate = frame.rename(columns={"U234_r": "U_234_r", "U234_r_err": "U_234_r_err",
                                         "Th230_r": "Th_230_r", "Th230_r_err": "Th_230_r_err",
                                         "Th232_r": "Th_232_r", "Th232_r_err": "Th_232_r_err"})
        rows = web._normalize_upload(alternate)
        self.assertEqual(len(rows), 10)
        self.assertEqual(rows[0]["Depth"], 6.0)
        self.assertAlmostEqual(rows[0]["Th232_r"], 0.002117)
        refs = [dict(sample=str(r.Sample), depth=float(r.Depth),
                     initial_th=float(r.Faraji_initial), sigma=float(r.Faraji_err))
                for r in frame.itertuples()]
        self.assertEqual(len(web.PriorRequest(rows=rows, reference=refs).reference), 10)
        with self.assertRaises(ValidationError):
            web.PriorRequest(rows=rows, reference=refs[:-1])

    def test_excel_workbook_parsing(self):
        # File shipped by the user is available during development but not in the ZIP.
        workbook = Path(web.__file__).parent.parent / "upload/Faraji_2021_data.xlsx"
        if workbook.exists():
            imported = web._normalize_upload(pd.read_excel(workbook, sheet_name=0))
            self.assertEqual(len(imported), 10)
            self.assertAlmostEqual(imported[6]["Th230_r"], 0.00477)

    def test_binary_excel_and_csv_import_endpoint(self):
        frame = pd.read_csv(Path(web.__file__).parent / "static/examples/faraji_2021.csv")

        async def import_data(content, filename):
            async def receive():
                return {"type": "http.request", "body": content, "more_body": False}
            scope = {"type": "http", "method": "POST", "path": "/api/import",
                     "query_string": f"filename={filename}".encode(), "headers": []}
            return await web.import_measurements(Request(scope, receive))

        excel = BytesIO()
        frame.to_excel(excel, index=False)
        for content, filename in ((excel.getvalue(), "measurements.xlsx"),
                                  (frame.to_csv(index=False).encode(), "measurements.csv")):
            imported = asyncio.run(import_data(content, filename))
            self.assertEqual(len(imported["rows"]), 10)
            self.assertEqual(imported["rows"][0]["Depth"], 6.0)

    def test_prior_and_mcmc(self):
        settings = web.Settings(n_samples_global=100, n_samples_layer=50,
                                n_chains=2, iterations=100, burn_in=50)
        request = web.PriorRequest(rows=example_rows(), settings=settings)
        prior_id = "smoke-prior"
        web._jobs[prior_id] = {"status": "running", "kind": "prior"}
        web._build_prior(prior_id, request)
        prior = web._jobs[prior_id]
        self.assertEqual(prior["status"], "complete")
        self.assertGreater(prior["result"]["prior_summary"]["median"], 0)

        run_id = "smoke-run"
        web._jobs[run_id] = {"status": "running", "kind": "part1"}
        web._run_part1(run_id, prior_id)
        result = web._jobs[run_id]["result"]
        self.assertEqual(len(result["rows"]), 4)
        for row in result["rows"]:
            self.assertGreater(row["ibis_age"], 0)
            self.assertGreater(row["ibis_hi95"], row["ibis_lo95"])
            self.assertGreater(row["uncorrected_sigma"], 0)
            self.assertGreater(row["bulk_sigma"], 0)


if __name__ == "__main__":
    unittest.main()
