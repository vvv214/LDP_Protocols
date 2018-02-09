# simplest test file, indicating build is success
from olh import generate

def test_olh():
    assert generate() < 1024
