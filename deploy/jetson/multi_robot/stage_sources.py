"""Create a small source-only archive for both Jetsons; no SSH or deployment."""
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[3]


def main():
    output = Path(__file__).resolve().parent / 'output/dpvo-online-sources.tar.gz'
    output.parent.mkdir(parents=True, exist_ok=True)
    excluded = {'.git', '__pycache__', 'build', 'dist', 'models', 'output',
                'bundle', 'agent-bundle', 'network-bin', 'sources', '.pixi', '.cache'}
    with tarfile.open(output, 'w:gz') as archive:
        for name in ('setup.py', 'dpvo', 'config', 'calib', 'ros2',
                     'deploy/jetson', 'DBoW2', 'DPRetrieval'):
            root = ROOT / name
            files = [root] if root.is_file() else sorted(root.rglob('*'))
            for path in files:
                relative = path.relative_to(ROOT)
                if (not path.is_file() or path.is_symlink()
                        or any(part in excluded or part.endswith('.egg-info') for part in relative.parts)
                        or path.suffix in {'.so', '.pyc', '.pth', '.engine'}):
                    continue
                archive.add(path, arcname=str(relative), recursive=False)
    print(output)


if __name__ == '__main__':
    main()
