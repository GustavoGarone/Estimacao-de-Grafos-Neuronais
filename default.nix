with import <nixpkgs> {};
  stdenv.mkDerivation {
    name = "Julia for NeuroMAT";
    buildInputs = [
      gcc
      stdenv.cc.cc.lib
    ];
    shellHook = ''
      export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
    '';
  }
