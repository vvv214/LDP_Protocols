# simplest test file, indicating build is success
import olh

def test_olh():
    assert olh.generate() < 1024
