class DataFix:
    """
    Represents a single fix recommendation for a data quality issue.
    Each fix includes metadata about its impact, risk, and implementation details.
    """

    def __init__(
        self,
        fix_id,
        issue_id,
        column,
        fix_label,
        fix_description,
        impact,
        risk,
        is_recommended=False,
        requires_user_input=False,
        metadata=None
    ):
        self.fix_id = fix_id
        self.issue_id = issue_id
        self.column = column
        self.fix_label = fix_label
        self.fix_description = fix_description
        self.impact = impact
        self.risk = risk
        self.is_recommended = is_recommended
        self.requires_user_input = requires_user_input
        self.metadata = metadata if metadata else {}

    def to_dict(self):
        """
        Convert fix object to dictionary format for easy serialization.
        """
        return {
            "fix_id": self.fix_id,
            "issue_id": self.issue_id,
            "column": self.column,
            "fix_label": self.fix_label,
            "fix_description": self.fix_description,
            "impact": self.impact,
            "risk": self.risk,
            "is_recommended": self.is_recommended,
            "requires_user_input": self.requires_user_input,
            "metadata": self.metadata
        }

    def __repr__(self):
        """
        String representation for debugging.
        """
        rec_tag = " [RECOMMENDED]" if self.is_recommended else ""
        return f"<DataFix: {self.fix_label}{rec_tag}>"