import json
from pathlib import Path
import pytest
from backend.evidence import claims, numeric_support
from backend.calibration import validate_profile


def test_abbreviations_preserve_whole_claim():
    assert claims('Dr. Reed opens the lab. Gold is yellow.')==['Dr. Reed opens the lab.','Gold is yellow.']


@pytest.mark.parametrize('claim,context,expected',[
    ('It opens at 9:30.','It opens at 09:30.',True),
    ('It opens at 09:30.','It opens at 09:00 and closes at 08:30.',False),
    ('The project has 1,000 records.','The project has 1000 records.',True),
    ('It launched in 2025.','It launched in 2024.',False),
])
def test_numeric_guard(claim,context,expected):
    assert numeric_support(claim,context)==expected


@pytest.mark.parametrize('field,value', [('weights',[float('nan'),0,0]),('scale',[0,1]),('threshold',2),('heads',[1.5])])
def test_invalid_fitted_profiles_fail_early(field,value):
    profile=json.loads((Path(__file__).resolve().parents[1]/'artifacts/calibration.json').read_text())
    profile[field]=value
    with pytest.raises(ValueError):validate_profile(profile)
