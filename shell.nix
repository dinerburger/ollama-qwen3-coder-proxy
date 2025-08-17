{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  name = "ollama-qwen3-proxy";

  buildInputs = with pkgs; [
    python3
    python3Packages.fastapi
    python3Packages.httpx
    python3Packages.uvicorn
    python3Packages.pydantic
  ];
}
