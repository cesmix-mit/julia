# Tapir-enabled `julia`

Building tapir-enabled `julia` requires building the Tapir/LLVM
compiler maintained by [OpenCilk project](http://cilk.mit.edu/).  Use
the following command to build Tapir/LLVM and Julia:

```sh
make USE_TAPIR=1
```
