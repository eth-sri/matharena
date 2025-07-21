from matharena.code_execution import CodeRunner

def test_basic():
    """Basic test that prints 'Hello, World!'."""
    code_runner = CodeRunner()
    result = code_runner.execute_python_code("print('Hello, World!')")
    code_runner.terminate()
    assert result["stdout"] == "Hello, World!\n"
    assert result["stderr"] == ""

def test_cpp_basic():
    """Basic test that prints 'Hello, World!'"""
    code_runner = CodeRunner()
    result = code_runner.execute_cpp_code("""#include<iostream>\nint main() { std::cout << "Hello, World!" << std::endl; return 0; }""")
    code_runner.terminate()
    assert result["stdout"] == "Hello, World!\n"
    assert result["stderr"] == ""

def test_persistency():
    """First execute code that writes an integer to a file, then execute code that reads the integer from the file and prints it + 10."""
    code_runner = CodeRunner()
    result1 = code_runner.execute_python_code("""
with open('file.txt', 'w') as f:
    f.write('42')
print("Integer written to file.txt")
""")
    assert result1["stdout"] == "Integer written to file.txt\n"
    assert result1["stderr"] == ""
    
    # Second code: Read the integer from file.txt and print it + 10
    result2 = code_runner.execute_python_code("""
with open('file.txt', 'r') as f:
    num = int(f.read())
print(num + 10)
""")
    code_runner.terminate()
    assert result2["stdout"] == "52\n"
    assert result2["stderr"] == ""

def test_libraries():
    """Test that the libraries are installed correctly."""
    code_runner = CodeRunner()
    result = code_runner.execute_python_code("import numpy as np; print(np.sum(np.array([1, 2, 3])))")
    code_runner.terminate()
    assert result["stdout"] == "6\n"
    assert result["stderr"] == ""

def test_time():
    """Test that the time is measured correctly."""
    code_runner = CodeRunner()
    result = code_runner.execute_python_code("import time; time.sleep(10); print('Hello, World!')")
    code_runner.terminate()
    assert result["stdout"] == "Hello, World!\n"
    assert result["stderr"] == ""
    assert result["time"] >= 9 and result["time"] <= 11
