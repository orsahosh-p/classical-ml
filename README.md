# classical-ml
Classical Machine Learning algorithms implemented from scratch in C17.

## Tests

Run all tests from the project directory:

```sh
./mltest
```

The script discovers `tests/test_*.c` recursively, compiles each test with
the implemented sources using C17 and strict warnings, and runs every test.
Warnings, compilation errors, and failed tests produce a nonzero exit code.
Temporary binaries are removed automatically. A C compiler is required;
set `CC` to choose a compiler (for example, `CC=clang ./mltest`).

To use `mltest` directly in the current shell:

```sh
export PATH="$PWD:$PATH"
mltest
```
