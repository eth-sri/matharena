"""Disjoint source ownership for the repository-token classification.

The graph is a declaration/dependency index, not a partition of source text:
structure fields overlap their parent, and commands such as ``recall`` and
``variable`` create no graph node. Source-only declarations may also live in
modules outside the imported graph. Count the benchmark's whole-file-masked
lines once, then classify source commands separately from declaration bodies.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, replace
import re

from leanlean.metrics.tokens import remove_lean_comments

# These boundaries are only needed for conventional declarations whose opening
# line has no graph owner. Graph-backed declarations retain the recorded spans.
_COMMAND = re.compile(
    r'^(?P<indent>[ \t]*)(?:@\[[^\n]*\]\s*)*'
    r'(?:(?:public|private|protected|noncomputable|unsafe|partial|local|scoped)\s+)*'
    r'(?P<kind>theorem|lemma|def|abbrev|opaque|axiom|constant|class|structure|inductive|instance|example|'
    r'namespace|section|end|variable|variables|universe|universes|open|export|attribute|include|omit|'
    r'set_option|recall|syntax|macro_rules|macro|elab_rules|elab|notation|infixl|infixr|infix|prefix|postfix|'
    r'initialize|builtin_initialize|mutual|import|module|suppress_compilation|recommended_spelling|'
    r'add_aesop_rules|register_simp_attr)\b|^(?P<hash_indent>[ \t]*)#\w+'
)
_DECLARATIONS = {'theorem','lemma','def','abbrev','opaque','axiom','constant','class','structure','inductive','instance','example'}
_NAME = re.compile(r'\s*(«[^»]+»|[\w.\'₀-₉]+)')


def with_source_only_declarations(graph, sources):
    """Index unrepresented declaration bodies without inventing dependency edges.

    Their identities include the source path: a duplicate theorem in a module
    outside the imported graph is distinct from the compiled theorem of the
    same name. No dependency edges are inferred for them. Retained bodies use
    the ordinary syntax/automation/proof ledger.
    """
    nodes = dict(graph.nodes)
    covered = collections.defaultdict(set)
    for node in nodes.values():
        path = sources.path_of_module(str(node.get('module','')))
        if path:
            covered[path].update(range(int(node['start_line'])-1,int(node['end_line'])))
    for path in sources.paths():
        lines = remove_lean_comments(sources._files[path],mask_strings=True).splitlines()
        commands = [(i,m) for i,line in enumerate(lines) if (m := _COMMAND.match(line))]
        occurrences = collections.Counter()
        for pos,(start,match) in enumerate(commands):
            kind=match.group('kind')
            if kind not in _DECLARATIONS or start in covered[path]:
                continue
            indent=len(match.group('indent') or '')
            end=next((i for i,m in commands[pos+1:] if len(m.group('indent') or m.group('hash_indent') or '')<=indent),len(lines))
            named=_NAME.match(lines[start][match.end():])
            label=named.group(1) if named else kind
            occurrences[label]+=1
            name=f'__source_only__:{path}:{label}:{occurrences[label]}'
            # A literal dot in a filename is part of that path component,
            # not a Lean module separator (e.g. scratch/Solution.orig.lean).
            module='.'.join('«'+part+'»' if '.' in part else part for part in path[:-5].split('/'))
            nodes[name]=dict(name=name,module=module,kind='theorem' if kind in {'theorem','lemma','example'} else 'def',
                             start_line=start+1,end_line=end,source_only=True,source_command=kind)
            covered[path].update(range(start,end))
    return replace(graph,nodes=nodes)


@dataclass
class SourcePartition:
    tokens: dict[str,int]
    texts: dict[str,str]
    source_commands: int
    source_command_lines: list[dict]
    overlap_tokens: int
    total_tokens: int


def partition(graph,sources,prefer=frozenset()):
    """Each scored source line belongs to one outer declaration or a command.

    Aliases with the same range prefer a name retained on both sides. Enclosing
    declarations own their fields/local helpers, so nested graph nodes never
    charge their parent's text again. Comments are masked in whole-file context
    before slicing; zero-scored import lines are omitted as in the benchmark.
    """
    by_path=collections.defaultdict(lambda:collections.defaultdict(list))
    tokens={name:0 for name in graph.nodes}
    texts={}
    paths = set(sources.paths())
    for name,node in graph.nodes.items():
        path=sources.path_of_module(str(node.get('module','')))
        if path in paths:
            by_path[path][(int(node['start_line']),int(node['end_line']))].append(name)
    outside=[];overlap=0;total=0
    for path in sources.paths():
        line_tokens=sources.line_tokens(path)
        clean=remove_lean_comments(sources._files[path]).splitlines(keepends=True)
        owners=[None]*len(line_tokens)
        multiplicity=[0]*len(line_tokens)
        ranges=by_path[path]
        for (start,end),names in sorted(ranges.items(),key=lambda item:(-(item[0][1]-item[0][0]),item[0])):
            owner=min(names,key=lambda name:(name not in prefer,name))
            for i in range(max(0,start-1),min(end,len(line_tokens))):
                multiplicity[i]+=1
                if owners[i] is None:
                    owners[i]=owner
                    tokens[owner]+=line_tokens[i]
        total+=sum(line_tokens)
        overlap+=sum(t*(n-1) for t,n in zip(line_tokens,multiplicity) if n>1)
        for i,(value,owner) in enumerate(zip(line_tokens,owners)):
            if owner is None and value:
                outside.append(dict(path=path,line=i+1,tokens=value,text=clean[i].rstrip()))
        for (start,end),names in ranges.items():
            for name in names:
                texts[name]=''.join(clean[i] if owners[i]==name and line_tokens[i] else '\n'
                                     for i in range(max(0,start-1),min(end,len(clean))))
    command_tokens=sum(line['tokens'] for line in outside)
    assert sum(tokens.values())+command_tokens==total==sources.total_tokens()
    return SourcePartition(tokens,texts,command_tokens,outside,overlap,total)
