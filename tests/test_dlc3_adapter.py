import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from MAS import inferenceMice as adapter
from MAS.tracking_io import DAM, PUP, read_tracking_csv

class Compatibility(unittest.TestCase):
    def test_headers_and_shuffled_columns(self):
        for depth in (3,4):
            with self.subTest(depth=depth), tempfile.TemporaryDirectory() as d:
                tuples=[(('scorer','dam') if depth==4 else ('scorer',))+(part,c)
                        for part in DAM+PUP for c in ('x','y','likelihood')]
                names=['scorer','individuals','bodyparts','coords'] if depth==4 else ['scorer','bodyparts','coords']
                data=pd.DataFrame(np.arange(78).reshape(2,39),columns=pd.MultiIndex.from_tuples(tuples,names=names))
                p=Path(d)/'track.csv';data.iloc[:,::-1].to_csv(p)
                result=read_tracking_csv(p)
                np.testing.assert_array_equal(result.iloc[:,1:].to_numpy(),data.to_numpy())
                data.index=[0,2];data.to_csv(p)
                with self.assertRaisesRegex(ValueError,'frame indices'):read_tracking_csv(p)
    def test_analysis_api_and_error_propagation(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)/'videos';folder.mkdir();(folder/'a.mp4').touch()
            config=Path(d)/'config.yaml';config.touch();api=Mock()
            with patch.object(adapter,'DLCDETECTOR',str(config)),patch.object(adapter,'_dlc',return_value=api):
                adapter.inference_mice(str(folder))
                kwargs=api.analyze_videos.call_args.kwargs
                self.assertEqual(kwargs['videos'],[str((folder/'a.mp4').resolve())])
                self.assertTrue(kwargs['auto_track'])
                self.assertTrue(Path(kwargs['destfolder']).is_dir())
                self.assertNotIn('engine',kwargs)
                api.analyze_videos.side_effect=OSError('model failure')
                with self.assertRaises(OSError):adapter.inference_mice(str(folder))
if __name__=='__main__':unittest.main()
