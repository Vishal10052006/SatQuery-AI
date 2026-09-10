"""
SatQuery AI - M4 Query Parser

Converts a natural-language satellite query into a structured
ParsedQuery object.

The parser is intentionally deterministic for the SIH MVP.
This gives us predictable routing and makes the system easy
to test within the limited development window.

Pipeline:

    Natural Language Query
            ↓
        QueryParser
            ↓
        ParsedQuery
"""


from app.query.schemas import Intent, Operation, ParsedQuery


class QueryParser:
    """
    Deterministic parser for SatQuery user queries.

    Responsibilities:
        1. Normalize the query.
        2. Detect the requested intent.
        3. Detect the requested operation.
        4. Extract the target object.
        5. Detect temporal comparison.
        6. Detect required modalities.
        7. Determine whether grounding/geospatial processing is needed.
    """

    # --------------------------------------------------------
    # Keyword groups
    # --------------------------------------------------------

    VQA_KEYWORDS = (
        "describe",
        "what is in",
        "what are in",
        "what do you see",
        "identify",
        "land cover",
        "scene",
        "image",
    )

    CHANGE_KEYWORDS = (
        "change",
        "changed",
        "difference",
        "differences",
        "between",
        "before and after",
        "compare",
        "comparison",
    )

    NEW_CHANGE_KEYWORDS = (
        "new",
        "newly constructed",
        "new construction",
        "constructed",
        "construction",
        "built",
        "appeared",
        "added",
    )

    GROUNDING_KEYWORDS = (
        "where",
        "location",
        "locate",
        "highlight",
        "show me",
        "region",
        "regions",
        "area",
    )

    OPTICAL_SAR_KEYWORDS = (
        "optical",
        "sar",
        "radar",
        "synthetic aperture",
    )

    TEMPORAL_KEYWORDS = (
        "between",
        "before",
        "after",
        "change",
        "changed",
        "over time",
        "temporal",
        "two images",
        "two dates",
    )

    # --------------------------------------------------------
    # Target vocabulary
    # --------------------------------------------------------

    TARGETS = (
        "buildings",
        "building",
        "roads",
        "road",
        "water",
        "water bodies",
        "water body",
        "vegetation",
        "deforestation",
        "forest",
        "farmland",
        "agriculture",
        "urban area",
        "built-up area",
        "built up area",
        "ships",
        "vehicles",
        "houses",
        "houses",
        "bridges",
        "land cover",
    )

    def parse(
        self,
        query: str,
        images: list[str] | None = None,
    ) -> ParsedQuery:
        """
        Parse a natural-language query into a ParsedQuery.

        Args:
            query:
                Natural-language user query.

            images:
                Optional image paths/identifiers. The parser uses
                this information to improve temporal detection.

        Returns:
            ParsedQuery:
                Structured representation of the query.

        Raises:
            ValueError:
                If the query is empty or contains only whitespace.
        """

        # ----------------------------------------------------
        # Validate input.
        # ----------------------------------------------------

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        # ----------------------------------------------------
        # Normalize the query.
        # ----------------------------------------------------

        normalized = self._normalize(query)

        # ----------------------------------------------------
        # Detect intent.
        # ----------------------------------------------------

        intent = self._detect_intent(normalized)

        # ----------------------------------------------------
        # Detect operation.
        # ----------------------------------------------------

        operation = self._detect_operation(
            normalized=normalized,
            intent=intent,
        )

        # ----------------------------------------------------
        # Extract target object.
        # ----------------------------------------------------

        target = self._extract_target(normalized)

        # ----------------------------------------------------
        # Detect temporal requirement.
        # ----------------------------------------------------

        temporal = self._detect_temporal(
            normalized=normalized,
            images=images,
        )

        # ----------------------------------------------------
        # Detect modality requirements.
        # ----------------------------------------------------

        modalities = self._detect_modalities(normalized)

        # ----------------------------------------------------
        # Detect grounding requirement.
        # ----------------------------------------------------

        requires_grounding = self._requires_grounding(
            normalized=normalized,
            intent=intent,
            operation=operation,
        )

        # ----------------------------------------------------
        # Geospatial processing is required when the query
        # explicitly requests location/regions or when
        # grounding is required for a spatial result.
        # ----------------------------------------------------

        requires_geospatial = requires_grounding

        # ----------------------------------------------------
        # Store additional extracted information.
        # ----------------------------------------------------

        entities = {
            "image_count": len(images or []),
        }

        return ParsedQuery(
            original_query=query,
            intent=intent,
            operation=operation,
            target=target,
            temporal=temporal,
            requires_grounding=requires_grounding,
            requires_geospatial=requires_geospatial,
            modalities=modalities,
            entities=entities,
        )

    # --------------------------------------------------------
    # Normalization
    # --------------------------------------------------------

    @staticmethod
    def _normalize(query: str) -> str:
        """
        Normalize user text for reliable keyword matching.
        """

        return " ".join(query.lower().strip().split())

    # --------------------------------------------------------
    # Intent detection
    # --------------------------------------------------------

    def _detect_intent(self, query: str) -> Intent:
        """
        Determine the primary SatQuery intent.

        Priority:

            Optical + SAR
                ↓
            Change Detection
                ↓
            Grounding
                ↓
            VQA
        """

        # Optical + SAR gets highest priority because it is
        # explicitly multimodal.
        has_optical = "optical" in query
        has_sar = (
            "sar" in query
            or "radar" in query
            or "synthetic aperture" in query
        )

        if has_optical and has_sar:
            return Intent.OPTICAL_SAR

        # Bi-temporal/change queries go to M2.
        if self._contains_any(query, self.CHANGE_KEYWORDS):
            return Intent.CHANGE_DETECTION

        # Explicit spatial grounding goes to M2 grounding.
        if self._contains_any(query, self.GROUNDING_KEYWORDS):
            return Intent.GROUNDING

        # Default visual understanding goes to M1.
        return Intent.VQA

    # --------------------------------------------------------
    # Operation detection
    # --------------------------------------------------------

    def _detect_operation(
        self,
        normalized: str,
        intent: Intent,
    ) -> Operation:
        """Determine the specific operation requested."""

        if intent == Intent.OPTICAL_SAR:
            return Operation.ANALYZE

        if intent == Intent.GROUNDING:
            return Operation.LOCALIZE

        if intent == Intent.CHANGE_DETECTION:

            if self._contains_any(
                normalized,
                self.NEW_CHANGE_KEYWORDS,
            ):
                return Operation.DETECT_NEW

            return Operation.DETECT_CHANGE

        if self._contains_any(
            normalized,
            ("describe", "what do you see", "what is in"),
        ):
            return Operation.DESCRIBE

        return Operation.ANSWER

    # --------------------------------------------------------
    # Target extraction
    # --------------------------------------------------------

    def _extract_target(self, query: str) -> str | None:
        """Extract the most relevant known satellite target.

        Targets are matched as complete words/phrases rather than
        arbitrary substrings. This prevents false matches such as
        ``forest`` inside ``deforestation``.
        """

        import re

        # Longer phrases must be checked first.
        ordered_targets = sorted(
            set(self.TARGETS),
            key=len,
            reverse=True,
        )

        for target in ordered_targets:
            pattern = rf"(?<!\\w){re.escape(target)}(?!\\w)"

            if re.search(pattern, query):
                return target

        return None

    # --------------------------------------------------------
    # Temporal detection
    # --------------------------------------------------------

    def _detect_temporal(
        self,
        normalized: str,
        images: list[str] | None,
    ) -> bool:
        """
        Determine whether the query requires temporal comparison.

        Two or more supplied images also strongly indicate
        a bi-temporal/multi-image request.
        """

        if len(images or []) >= 2:
            return True

        return self._contains_any(
            normalized,
            self.TEMPORAL_KEYWORDS,
        )

    # --------------------------------------------------------
    # Modality detection
    # --------------------------------------------------------

    def _detect_modalities(self, query: str) -> list[str]:
        """Extract requested image modalities."""

        modalities: list[str] = []

        if "optical" in query:
            modalities.append("optical")

        if (
            "sar" in query
            or "radar" in query
            or "synthetic aperture" in query
        ):
            modalities.append("sar")

        # If no modality is explicitly mentioned, assume
        # generic optical imagery for the MVP.
        if not modalities:
            modalities.append("optical")

        return modalities

    # --------------------------------------------------------
    # Grounding detection
    # --------------------------------------------------------

    def _requires_grounding(
        self,
        normalized: str,
        intent: Intent,
        operation: Operation,
    ) -> bool:
        """
        Determine whether the result should identify a region
        or spatial location.
        """

        if intent == Intent.GROUNDING:
            return True

        if operation == Operation.LOCALIZE:
            return True

        if self._contains_any(
            normalized,
            (
                "highlight",
                "where",
                "locate",
                "show the region",
                "show regions",
                "where did the change occur",
            ),
        ):
            return True

        # Targeted new construction queries generally benefit
        # from spatial grounding.
        if (
            operation == Operation.DETECT_NEW
            and self._extract_target(normalized) is not None
        ):
            return True

        return False

    # --------------------------------------------------------
    # Utility
    # --------------------------------------------------------

    @staticmethod
    def _contains_any(
        query: str,
        keywords: tuple[str, ...],
    ) -> bool:
        """Return True if any keyword occurs in the query."""

        return any(keyword in query for keyword in keywords)


# ------------------------------------------------------------
# Default parser instance
# ------------------------------------------------------------

parser = QueryParser()
