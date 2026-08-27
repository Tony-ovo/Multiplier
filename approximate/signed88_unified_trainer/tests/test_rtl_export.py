import tempfile,unittest
from pathlib import Path
from signed88.hardware import get_design
from signed88.common import read_json
ROOT=Path(__file__).resolve().parents[1]
class RtlExportTest(unittest.TestCase):
 def test_export_all(self):
  for name in ['aggressive','fast','default','default_split','balanced','balanced_split','quality','area']:
   with self.subTest(name=name), tempfile.TemporaryDirectory() as td:
    d=get_design(name);out=d.export_rtl(ROOT/'rtl_sources',Path(td)/name,d.spec.base_inits);obj=read_json(out/'trained_artifact.json');self.assertEqual(obj['design'],name);self.assertEqual(d.normalize_inits(obj['inits']),d.normalize_inits(d.spec.base_inits))
if __name__=='__main__':unittest.main()
