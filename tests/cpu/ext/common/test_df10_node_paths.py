"""DF10 (2026-10-24): ENV-C's locked pair on WINH2, where /data is a symlink to /mnt/data: root's reader refused every
candidate file ("out/run_meta.json is not inside out") - the file's path was taken relative to the root's resolved
path. A file named under the root as the caller names it is inside it; a symlink below the root is still refused."""
import os

import pytest

from infrarsi.ext.common import node_ext

pytestmark = pytest.mark.skipif(not hasattr(os, 'O_NOFOLLOW'), reason='the node side runs on Linux')


def tree(tmp_path):
    real = tmp_path / 'mnt' / 'data' / 'runs' / 'out'
    real.mkdir(parents=True)
    (real / 'run_meta.json').write_text('{"ok": 1}')
    (tmp_path / 'data').symlink_to(tmp_path / 'mnt' / 'data')
    return tmp_path / 'data' / 'runs' / 'out', real


def test_a_file_under_a_root_reached_through_a_symlinked_mount_is_inside_it(tmp_path):
    out, real = tree(tmp_path)
    assert node_ext.read_user_file(str(out / 'run_meta.json'), str(out))['ok']
    assert node_ext.read_user_file(str(real / 'run_meta.json'), str(out))['ok']          # named by its resolved path


def test_a_link_below_the_root_and_a_path_outside_are_still_refused(tmp_path):
    out, real = tree(tmp_path)
    (real / 'evil').symlink_to(tmp_path / 'mnt' / 'data' / 'runs')
    (real / 'sub').mkdir()
    (real / 'sub' / 'f').write_text('x')
    os.symlink(real / 'sub', real / 'linkdir')
    assert not node_ext.read_user_file(str(out / 'evil'), str(out))['ok']                # the file is a link
    assert not node_ext.read_user_file(str(out / 'linkdir' / 'f'), str(out))['ok']      # a directory link below the root
    with pytest.raises(ValueError):
        node_ext.read_user_file(str(tmp_path / 'data' / 'runs' / 'x'), str(out))         # outside
