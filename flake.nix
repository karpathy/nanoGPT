{
  description = "NanoGPT - The simplest, fastest repository for training/finetuning medium-sized GPTs";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };

        python = pkgs.python313;
        pythonPackages = python.pkgs;

        pythonEnv = python.withPackages (
          ps: with ps; [
            torch
            numpy
            transformers
            datasets
            tiktoken
            wandb
            tqdm
          ]
        );

        cudaLibs = with pkgs; [
          cudatoolkit
          cudaPackages.cudnn
          cudaPackages.nccl
        ];
        systemLibs = with pkgs; [
          glibc
          gcc-unwrapped.lib
          libgcc.lib
          openblas
          lapack
          gfortran.cc.lib
          hdf5
          szip
          libaec
          libjpeg
          libpng
          libtiff
          freetype
          harfbuzz
          lcms2
          openjpeg
          libwebp
          brotli
          xz
          xorg.libX11
          xorg.libXau
          xorg.libxcb
          zeromq
          libsodium
          zlib
          openssl
          pkg-config
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = systemLibs ++ cudaLibs;
          packages = [ pythonEnv ];
          env = {
            LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath (systemLibs ++ cudaLibs)}";
            CUDA_PATH = "${pkgs.cudatoolkit}";
            CUDA_ROOT = "${pkgs.cudatoolkit}";
            CUDNN_PATH = "${pkgs.cudaPackages.cudnn}";
            XLA_FLAGS = "--xla_gpu_cuda_data_dir=${pkgs.cudatoolkit}/lib";
            PKG_CONFIG_PATH = "${pkgs.lib.makeSearchPathOutput "dev" "lib/pkgconfig" (systemLibs ++ cudaLibs)}";
            NIX_CFLAGS_COMPILE = "-I${pkgs.lib.makeSearchPathOutput "dev" "include" (systemLibs ++ cudaLibs)}";
            NIX_LDFLAGS = "-L${pkgs.lib.makeLibraryPath (systemLibs ++ cudaLibs)}";
          };
        };
      }
    );
}
