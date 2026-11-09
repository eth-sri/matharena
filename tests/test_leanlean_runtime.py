"""Keep the default stack reservation bounded without overriding agent flags."""

import pytest

from leanlean.lean_runtime import command


@pytest.mark.parametrize('arguments', [[], ['--run', 'Main.lean'], ['Main.lean', '-DmaxRecDepth=30000']])
def test_lean_stack_default(arguments):
    assert command('/bin/.lean-real', arguments) == ['/bin/.lean-real', '--tstack=262144', *arguments]


@pytest.mark.parametrize('arguments', [
    ['-s', '65536', 'Main.lean'], ['-s65536', 'Main.lean'],
    ['--tstack', '65536', 'Main.lean'], ['--tstack=65536', 'Main.lean'],
])
def test_explicit_lean_stack_is_preserved(arguments):
    assert command('/bin/.lean-real', arguments) == ['/bin/.lean-real', *arguments]
