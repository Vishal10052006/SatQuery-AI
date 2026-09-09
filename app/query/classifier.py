"""
SatQuery AI - M4 Intent Classifier

Maps a structured ParsedQuery to the specialist module/tool
responsible for executing that query.

Pipeline:

    Natural Language Query
            ↓
        QueryParser
            ↓
        ParsedQuery
            ↓
        IntentClassifier
            ↓
        ToolName
"""


from app.query.schemas import Intent, ParsedQuery, ToolName


class IntentClassifier:
    """
    Deterministic intent-to-tool classifier.

    The parser is responsible for understanding the query.
    This classifier is responsible for deciding which specialist
    module should execute it.
    """

    # --------------------------------------------------------
    # Intent → Tool mapping
    # --------------------------------------------------------

    INTENT_TO_TOOL: dict[Intent, ToolName] = {
        # M1 handles visual question answering and
        # single-image visual understanding.
        Intent.VQA: ToolName.M1_VQA,

        # M2 handles bi-temporal change detection.
        Intent.CHANGE_DETECTION: ToolName.M2_CHANGE_DETECTION,

        # M2 also handles spatial/referring grounding.
        Intent.GROUNDING: ToolName.M2_GROUNDING,

        # M3 handles optical + SAR multimodal analysis.
        Intent.OPTICAL_SAR: ToolName.M3_OPTICAL_SAR,
    }

    def classify(self, parsed_query: ParsedQuery) -> ToolName:
        """
        Select the specialist tool for a parsed query.

        Args:
            parsed_query:
                Structured query produced by QueryParser.

        Returns:
            ToolName:
                Specialist tool responsible for the request.

        Raises:
            ValueError:
                If an unsupported intent is encountered.
        """

        try:
            return self.INTENT_TO_TOOL[parsed_query.intent]

        except KeyError as exc:
            raise ValueError(
                f"Unsupported SatQuery intent: {parsed_query.intent}"
            ) from exc


# ------------------------------------------------------------
# Default classifier instance
# ------------------------------------------------------------

classifier = IntentClassifier()
