"""Fail-closed structural allowlist for a deliberately small Cypher language.

This is not a general Cypher parser. It lexes identifiers/parameters/operators,
parses balanced expression groups into an immutable tree, then accepts only
registered template trees. Unsupported syntax, comments, quoted literals,
procedures, UNION, and extra statements are rejected rather than sanitized.
"""

from dataclasses import dataclass

from fingraph_rag.domain.errors import PolicyViolation
from fingraph_rag.graph.cypher_templates import TRUSTED_READ_QUERIES


@dataclass(frozen=True)
class Group:
    delimiter: str
    children: tuple["str | Group", ...]


@dataclass(frozen=True)
class ParsedReadQuery:
    tree: Group
    parameters: frozenset[str]


def _parse(query: str) -> ParsedReadQuery:
    if not isinstance(query, str) or not query.strip() or len(query) > 8192:
        raise PolicyViolation("invalid Cypher length")
    tokens: list[str] = []
    index = 0
    while index < len(query):
        char = query[index]
        if char.isspace():
            index += 1
            continue
        if char.isascii() and (char.isalpha() or char in "_$"):
            end = index + 1
            while (
                end < len(query)
                and query[end].isascii()
                and (query[end].isalnum() or query[end] == "_")
            ):
                end += 1
            token = query[index:end]
            if token == "$":
                raise PolicyViolation("empty query parameter")
            tokens.append(token)
            index = end
        elif char.isascii() and char.isdigit():
            end = index + 1
            while end < len(query) and query[end].isascii() and query[end].isdigit():
                end += 1
            tokens.append(query[index:end])
            index = end
        elif char in "()[]:,.=*|-<>+":
            if query[index : index + 2] in {"//", "/*", "--"}:
                raise PolicyViolation("comments and unsupported operators are forbidden")
            tokens.append(char)
            index += 1
        else:
            raise PolicyViolation("unsupported Cypher syntax")
    stack: list[tuple[str, list[str | Group]]] = [("root", [])]
    close = {")": "(", "]": "["}
    for token in tokens:
        if token in {"(", "["}:
            stack.append((token, []))
        elif token in close:
            if len(stack) == 1 or stack[-1][0] != close[token]:
                raise PolicyViolation("unbalanced Cypher expression")
            delimiter, children = stack.pop()
            stack[-1][1].append(Group(delimiter, tuple(children)))
        else:
            stack[-1][1].append(token)
    if len(stack) != 1:
        raise PolicyViolation("unbalanced Cypher expression")
    return ParsedReadQuery(
        Group("root", tuple(stack[0][1])),
        frozenset(token[1:] for token in tokens if token.startswith("$")),
    )


class CypherAstPolicy:
    """Only immutable project-owned read templates may cross the adapter."""

    def __init__(self) -> None:
        self._allowed = frozenset(_parse(query).tree for query in TRUSTED_READ_QUERIES)

    def validate(self, query: str) -> ParsedReadQuery:
        parsed = _parse(query)
        if parsed.tree not in self._allowed:
            raise PolicyViolation("Cypher AST is outside the registered read-template allowlist")
        return parsed
