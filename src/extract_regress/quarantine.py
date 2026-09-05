"""Deferring a fixture you already know is broken, with an expiry.

`run` fails the build on any regression, which is the point, and leaves nowhere
to put a fixture that is broken for a reason nobody is fixing today: a vendor
changed their PDF layout, a model was deprecated, a field needs a new prompt.
The alternatives are all bad. Left failing, the build is red and within a week
nobody reads it. Deleted, the fixture, its coverage row and the record that it
used to work all go with it. Filtered out with ``-k``, the exclusion lives in
whatever CI YAML someone edited, invisible to the next person.

A quarantine entry is a written decision to defer, and it has teeth so that it
stays one:

* A quarantined fixture that **passes** fails the build. A stale quarantine is
  worse than none, because it silently swallows the regression when it returns.
* ``until`` is required and enforced. A deferral with no end is a deletion
  nobody wrote down.
* ``reason`` is required. An entry with no reason is a mystery to whoever finds
  it in six months.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, field_validator


class QuarantineRule(BaseModel):
    """One deferred fixture."""

    model_config = ConfigDict(frozen=True)

    #: The fixture name. Exact, not a glob: a pattern would let one entry
    #: quietly cover fixtures added later.
    name: str
    #: Why it is deferred. Required.
    reason: str
    #: The day the deferral runs out, inclusive. Required.
    until: date

    @field_validator("name", "reason")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value

    def expired(self, today: date) -> bool:
        """Whether the deferral has run out as of ``today``."""
        return today > self.until


class QuarantineConfig(BaseModel):
    """The quarantine list for a project."""

    model_config = ConfigDict(frozen=True)

    rules: tuple[QuarantineRule, ...] = Field(default_factory=tuple)

    @field_validator("rules")
    @classmethod
    def _unique_names(cls, rules: tuple[QuarantineRule, ...]) -> tuple[QuarantineRule, ...]:
        seen: set[str] = set()
        for rule in rules:
            if rule.name in seen:
                # Two entries for one fixture means two different stated
                # reasons and two different expiry dates, and no way to say
                # which one is the decision.
                raise ValueError(f"duplicate quarantine entry for fixture {rule.name!r}")
            seen.add(rule.name)
        return rules

    def rule_for(self, fixture_name: str) -> QuarantineRule | None:
        """The entry covering ``fixture_name``, or ``None``."""
        for rule in self.rules:
            if rule.name == fixture_name:
                return rule
        return None
