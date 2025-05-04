{
  description = "A Nix-flake-based Python development environment";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = {
    self,
    nixpkgs,
    ...
  }: let
    # system should match the system you are running on
    system = "x86_64-linux";
    # system = "x86_64-darwin";
  in {
    devShells."${system}".default = let
      pkgs = import nixpkgs {
        inherit system;
      };
    in
      pkgs.mkShell {
        # create an environment with nodejs-18_x, pnpm, and yarn
        packages = with pkgs; [
          (pkgs.python3.withPackages (python-pkgs:
            with python-pkgs; [
              # select Python packages here
              pandas
              requests
              matplotlib
              notebook
              jupyterlab
              numpy
              pip
              scipy
              pillow
              isort
              black
              flake8
              pylint
              mypy
            ]))
          pkgs.zlib
          pkgs.zstd
          pkgs.curl
          pkgs.openssl
          pkgs.attr
          pkgs.libssh
          pkgs.bzip2
          pkgs.libxml2
          pkgs.acl
          pkgs.libsodium
          pkgs.util-linux
          pkgs.xz
          pkgs.systemd
          pkgs.xorg.libX11
          pkgs.pdftk
        ];

        shellHook = ''
          echo "python `python --version`"
        '';
      };
  };
}
