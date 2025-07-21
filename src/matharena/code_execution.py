import modal
import time

PY_LIBRARIES = [
    "pandas", "numpy", "scikit-learn", "sympy", "gmpy2",
]

EXEC_TIMEOUT = 120

class CodeRunner:

    def __init__(self, sandbox_timeout=3600, n_retries=3):
        self.app = modal.App.lookup("project-euler-matharena", create_if_missing=True)
        self.sandbox = modal.Sandbox.create(
            image=modal.Image.debian_slim(python_version="3.12").pip_install(PY_LIBRARIES),
            app=self.app,
            timeout=sandbox_timeout,
            block_network=True,
        )
        self.n_exec = 0
        self.n_retries = n_retries

    def execute_python_code(self, code):
        """Writes Python code to a file, executes it and returns the result."""
        print("Executing code:\n", code)
        for _ in range(self.n_retries):
            try:
                filename = f"pycode_{self.n_exec}.py"

                f = self.sandbox.open(filename, "w")
                f.write(code)
                f.close()

                time_start = time.time()
                p = self.sandbox.exec("bash", "-c", f"python {filename}", timeout=EXEC_TIMEOUT)
                self.n_exec += 1

                output = {
                    "stdout": p.stdout.read(),
                    "stderr": p.stderr.read(),
                }
                output["time"] = time.time() - time_start
                print("code execution output: ", output)
                return output
            except Exception as e:
                print(f"Error executing code: {e}")
                time.sleep(1)
        raise Exception("Failed to execute code")

    def execute_cpp_code(self, code):
        """Writes C++ code to a file, compiles it and returns the result."""
        print("Executing C++ code:\n", code)
        for _ in range(self.n_retries):
            try:
                filename = f"cppcode_{self.n_exec}.cpp"

                f = self.sandbox.open(filename, "w")
                f.write(code)
                f.close()

                time_start = time.time()
                p = self.sandbox.exec("bash", "-c", f"g++ {filename} -o {filename}.out && ./{filename}.out", timeout=EXEC_TIMEOUT)
                time_end = time.time()
                self.n_exec += 1

                output = {
                    "stdout": p.stdout.read(),
                    "stderr": p.stderr.read(),
                }
                output["time"] = time.time() - time_start
                print("C++ code execution output: ", output)
                return output
            except Exception as e:
                print(f"Error executing code: {e}")
                time.sleep(1)
        raise Exception("Failed to execute code")

    def terminate(self):
        self.sandbox.terminate()
