{
  description = "Random walk benchmark dev shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs = inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];

      perSystem = { pkgs, ... }: {
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = with pkgs; [
            gcc
            ninja
            uv
          ];

          buildInputs = with pkgs; [
            stdenv.cc.cc.lib   # libgomp (OpenMP runtime)
          ];

          shellHook = ''
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"
          '';
        };
      };
    };
}
