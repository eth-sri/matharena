from dataclasses import dataclass
from types import SimpleNamespace

from leanlean.macro_driver import command_sites, preceding_doc, rule_head


def test_elaboration_rule_head_skips_optional_antiquotations():
    assert rule_head('$[$d]? slots $[$n]? $xs*') == 'slots'
    assert rule_head('$d:docComment slots $xs*') == 'slots'
    assert rule_head('$[$nested[$x]]* slots $xs*') == 'slots'
    assert rule_head('$[unfinished') is None


def test_documentation_prefix_respects_nested_comments_and_adjacency():
    text = '/-- outer /- inner -/ documentation -/\n  '
    assert preceding_doc(text, len(text)) == 0
    assert preceding_doc(text + 'def other := 1\n', len(text) + 15) is None


@dataclass
class Site:
    path: str
    start: int
    end: int
    text: str
    category: str = 'command'
    lead: str = 'v'


def test_documented_command_sites_include_the_required_doc_but_exclude_variables():
    body = 'notation "v" => fun v => v\n/-- Coordinates. -/\nv coords 1, 2\n'
    program = SimpleNamespace(lead='v', category='command', source='local macro d:docComment "v" i:ident x:term,* : command => pure default')
    starts = [body.index('v =>'), body.index('v coords')]
    sites = [Site('Main.lean', n, len(body), body[n:]) for n in starts]
    kept = command_sites(sites, [program], {'Main.lean': body})
    assert len(kept) == 1
    assert kept[0].text == '/-- Coordinates. -/\nv coords 1, 2\n'
    assert kept[0].start == body.index('/--')
