---
layout: post
title: Reproducible dev environments with Nix and Home Manager
date: 2026-08-09
description: How I stopped hand-configuring GCP instances and made my dev environment reproducible with Nix and Home Manager
slug: reproducible-dev-environments-with-nix
tags: machine-learning nix
categories: nix
featured: true
---

I have been training a bunch of models and running inference on Google Cloud Platform (GCP) 
for a hobby project. I don't get managed instances — this is a personal hobby project — 
so one big pain point is having my dev environment reproduced on the many instances I start.

GCP does provide the `--metadata-from-file` flag, which lets you run a shell script that sets things up on the machine at boot. 
I used that for a while, but was never satisfied: what ends up installed is not exactly the same each time, and that
kind of non-determinism gives me a failure mode that is hard to debug. An environment that is *almost* right 
sometimes causes more grief than one that is clearly broken — often giving me subtle bugs that take an awful lot of time to debug.


## Nix

Enter [Nix](https://nixos.org/learn/). Nix is a package manager that builds software from declarative 
descriptions, so the same configuration produces 
the same environment on any machine, regardless of what the host already has installed. 
[Home Manager](https://nix-community.github.io/home-manager/introduction.html) extends that idea to your home directory: it declaratively manages your shell, editor, and tools on Linux machines where you don't have `sudo` privileges[^1] — building everything into the Nix store and symlinking it into your home, so the environment is reproducible without touching the system.

Home Manager lets me declaratively install my shell, editor, git config, and tools. They are built into the Nix store (`/nix/store`) and symlinked into my home directory — so the whole environment is reproducible, and nothing outside my home is touched.


## Using this environment

```bash
# 1. install nix (Determinate Systems installer, no root needed)
curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh -s -- install

# 2. clone and activate (pick the arch matching `uname -m`)
git clone git@github.com:govinda-kamath/home-manager-dotfiles.git ~/home-manager-dotfiles
cd ~/home-manager-dotfiles
nix run --impure .#homeConfigurations.x86_64-linux.activationPackage

# 3. done — start the configured shell
exec zsh
```

First activation prompts for my git name/email (saved to `~/.gitconfig.local`) and generates an SSH key — 
add the printed public key to GitHub and you're done.


## What I put in

I largely use `zsh`, which I set up with standard plugins like autocompletion, autosuggestion, and syntax highlighting,
and configured my command line the way I like it (it shows my git branch and conda environment). 
I also set up standard tools for `python`, `rust` and `go` — the languages I like programming in — including `micromamba`,
which I use for maintaining environments. 
I also set up `neovim` with a configuration I like, with LSPs and `treesitter`.

I use [Ghostty](https://ghostty.org/), which needed some work: I had to set up `TERMINFO` for `xterm-ghostty`.

I also added some tools — many standard ones like `ripgrep`, `fd`, and `jq`, and some esoteric ones like `eza`.

## How it's structured

The whole environment lives in a [flake](https://github.com/govinda-kamath/home-manager-dotfiles/blob/main/flake.nix) — 
a self-contained Nix project pinned to `release-25.05` of both nixpkgs and home-manager, so the setup stays stable and reproducible, 
no surprise breaks when I come back to a machine months later. The flake builds the same config for `x86_64` and `aarch64`, and 
reads `$USER` at eval time (`--impure`), so the same repo works for any login name without editing anything.

The config is split into small modules, one per concern — shell, dev toolchains (python/rust/go), neovim, git, ssh, 
the ghostty terminal, and conda — which keeps each file short and easy to tweak. A `hosts/` directory holds per-machine overrides 
(work tools, extra packages) that are imported automatically on the machine whose hostname matches. 
Anything machine-specific or personal, like git identity, lives in gitignored files, so the repo itself stays clean and shareable.

One thing worth copying: because every machine imports the same modules, differences between instances are deliberate 
(in `hosts/`) rather than accidental — which is exactly the drift problem that motivated this whole setup.

## Things I learned

- **Installing Nix without `sudo` is not easy**: most of the setup after Nix is installed doesn't need `sudo`, but I wasn't able to install Nix itself without `sudo` permissions.
- **Home Manager owns your dotfiles**: `~/.zshrc`, `~/.bashrc`, and everything under `~/.config` become 
read-only symlinks into the Nix store, so you can't hand-edit them. This takes a bit of getting used to.
You also can't run things like `chsh` and `byobu-enable`, though there are easy workarounds. <!-- CHECK: this sentence was cut off in your draft — verify -->
- **Machine-specific bits stay out of git**: `~/.gitconfig.local` and `hosts/` are gitignored — per-machine 
secrets and tweaks live there, and the repo stays clean and shareable.
- **Terminal stuff over SSH is fiddly**: setting `TERMINFO_DIRS` makes terminals like Ghostty work over 
SSH without the remote having it installed system-wide — small config, big quality-of-life win.

The repo is [github.com/govinda-kamath/home-manager-dotfiles](https://github.com/govinda-kamath/home-manager-dotfiles) -- 
in case anyone wants to adapt it to their needs. Feedback/criticism welcome!!

[^1]: Installing Nix is much smoother with `sudo` privileges though.