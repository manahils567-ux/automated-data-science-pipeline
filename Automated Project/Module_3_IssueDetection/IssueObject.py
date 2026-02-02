class DataIssue:
    """
    Represents a single data quality issue found in the dataset.
    """
    def __init__(self, issue_id, column, issue_type, severity, description, examples):
        self.issue_id = issue_id
        self.column = column
        self.issue_type = issue_type
        self.severity = severity
        self.description = description
        self.examples = examples

    def to_dict(self):
        return {
            "issue_id": self.issue_id,
            "column": self.column,
            "issue_type": self.issue_type,
            "severity": self.severity,
            "description": self.description,
            "examples": self.examples
        }