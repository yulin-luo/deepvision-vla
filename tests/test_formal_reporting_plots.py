import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from deepvision_vla.diagnostics import formal_reporting as f
from test_formal_reporting import fixture


@unittest.skipUnless(importlib.util.find_spec('matplotlib'),'Existing figure runtime required; do not install into the model environment')
class ReportPlotTests(unittest.TestCase):
    def test_task_roi_and_semantic_controls_are_rendered(self):
        sources,requests,terminals,ref=fixture("openvla",random=False,semantic=True,primary_region=f.TASK_PRIMARY)
        result=f.build_model_report(sources,requests,terminals,"openvla",reference_sha=ref,
            semantic_requested=True,random_requested=False,primary_region=f.TASK_PRIMARY)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"task_roi_report"
            report=f.write_report(out,{"openvla":result},sources,admission={},input_seals={"synthetic_fixture_only":True})
            curves=json.loads((out/"figures/PLOT_CONTRACTS.json").read_text(encoding="utf-8"))
            self.assertEqual([row["group"] for row in curves].count("semantic"),3)
            self.assertEqual([row["group"] for row in curves].count("primary"),3)
            self.assertEqual(len(report["models"]["openvla"]["semantic_contrasts"]),3)

    def test_synthetic_all_models_json_csv_sources_png_pdf_and_no_overwrite(self):
        reports={};sources=None
        for model in f.MODELS:
            sources,requests,terminals,ref=fixture(model)
            reports[model]=f.build_model_report(sources,requests,terminals,model,reference_sha=ref)
            reports[model]['native_counters']=None
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'synthetic_report'
            report=f.write_report(out,reports,sources,admission={'synthetic_fixture_only':True},input_seals={'synthetic_fixture_only':True})
            self.assertEqual(report['requested_source_identities'],2)
            self.assertEqual(len((out/'source_coverage.jsonl').read_text(encoding='utf-8').splitlines()),6)
            contracts=json.loads((out/'figures/PLOT_CONTRACTS.json').read_text(encoding='utf-8'))
            self.assertEqual(len(contracts),24)
            for contract in contracts:
                stem=out/'figures'/contract['filename']
                self.assertTrue(stem.with_suffix('.png').read_bytes().startswith(b'\x89PNG'))
                self.assertTrue(stem.with_suffix('.pdf').read_bytes().startswith(b'%PDF'))
                self.assertEqual(contract['keep_ratios'],list(f.KEEPS))
                if 'layers' in contract:self.assertEqual(contract['layers'],list(f.LAYERS[contract['model']]))
            csv=(out/'curves.csv').read_text(encoding='utf-8')
            self.assertIn('pointwise95%_not_simultaneous',csv)
            self.assertIn('source_bound_H4_reference_mse',csv);self.assertIn('normalized_H16_prediction_drift',csv)
            self.assertIn('-0.',csv)
            manifest=json.loads((out/'ARTIFACT_MANIFEST.json').read_text(encoding='utf-8'))
            for item in manifest['files']:self.assertEqual(f.file_sha(out/item['path']),item['sha256'])
            with self.assertRaises(FileExistsError):f.write_report(out,reports,sources,admission={},input_seals={})

    def test_missing_panel_is_gap_not_zero_and_plot_contract_incomplete(self):
        from deepvision_vla.diagnostics.formal_reporting_plots import _values
        import numpy as np
        mean,lo,hi=_values({'means':[None,0.],'lower':[None,None],'upper':[None,None]},[0,1])
        self.assertTrue(np.isnan(mean[0]));self.assertEqual(mean[1],0.)
        self.assertTrue(np.isnan(lo).all());self.assertTrue(np.isnan(hi).all())


class AttributionProjectionTests(unittest.TestCase):
    def test_source_crop_and_model_padding_use_native_cell_coordinates(self):
        import numpy as np
        from deepvision_vla.diagnostics.attribution_plot import source_overlay
        from deepvision_vla.diagnostics.geometry import VisualLayout

        layout = VisualLayout(
            image_sha256="0" * 64,
            source_size=(8, 4),
            crop_xyxy=(2, 0, 6, 4),
            canvas_size=(8, 4),
            content_xyxy=(2, 0, 6, 4),
            grid_shape=(2, 4),
            token_positions=tuple(range(8)),
        )
        projected = source_overlay(np.arange(8).reshape(2, 4), layout)
        np.testing.assert_array_equal(projected, [
            [0, 0, 1, 1, 2, 2, 0, 0],
            [0, 0, 1, 1, 2, 2, 0, 0],
            [0, 0, 5, 5, 6, 6, 0, 0],
            [0, 0, 5, 5, 6, 6, 0, 0],
        ])



class AttributionScaleTests(unittest.TestCase):
    def test_scale_keeps_shared_strength_and_removes_flat_layer_contrast(self):
        import numpy as np
        from deepvision_vla.diagnostics.attribution_plot import normalized_positive_scores

        values = np.array([[0.2, 0.4], [0.6, 1.0]])
        shared = normalized_positive_scores(values, 2.0, "model")
        layer = normalized_positive_scores(values, 2.0, "layer")
        np.testing.assert_array_equal(shared, values / 2.0)
        np.testing.assert_allclose(layer, [[0.0, 0.25], [0.5, 1.0]])
        flat = np.full((2, 2), 0.8)
        np.testing.assert_array_equal(normalized_positive_scores(flat, 2.0, "layer"), np.zeros_like(flat))
        np.testing.assert_array_equal(flat, np.full((2, 2), 0.8))

if __name__=='__main__':unittest.main(verbosity=2)
