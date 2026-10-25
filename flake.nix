{
  description = "Rust development environment for lance_arxiv";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      protocWithIncludes = pkgs.writeShellScriptBin "protoc-with-includes" ''
        exec ${pkgs.protobuf}/bin/protoc -I${pkgs.protobuf}/include "$@"
      '';
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          pkgs.openssl
          pkgs.pkg-config
          pkgs.protobuf
          protocWithIncludes
        ];
        PROTOC = "${protocWithIncludes}/bin/protoc-with-includes";
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
          pkgs.openssl
          pkgs.stdenv.cc.cc.lib
        ];
      };
    };
}
