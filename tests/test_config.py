from annb.config import load_configs

def test_load_configs_foo(tmpdir):
    content = """
default:
  result: test

runs:
  - name: test1
  - name: test2
"""
    with open(tmpdir.join("test.yaml"), "w") as f:
        f.write(content)
    runs = load_configs(tmpdir.join("test.yaml"))
