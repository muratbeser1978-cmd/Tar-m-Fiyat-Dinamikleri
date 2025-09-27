"""Mathematical Extraction Orchestrator.

This module implements the ExtractionOrchestrator class that coordinates
all extraction components with constitutional mathematical fidelity requirements.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
import logging
import time
from pathlib import Path

from .parsers.latex_parser import LaTeXParser, ParsedExpression
from .extractors.equation_extractor import EquationExtractor, ExtractedEquation, ExtractionResult
from .validators.dependency_analyzer import DependencyAnalyzer, DependencyAnalysisResult, convert_role_to_variable_type
from .validators.sde_classifier import SDEClassifier, ClassificationResult
from ..core.equations.mathematical_equation import MathematicalEquation
from ..core.variables.mathematical_variable import MathematicalVariable
from ..core.equations.dependency_graph import DependencyGraph
from ..sde.models.sde_model import SDEModel

# Configure logging
logger = logging.getLogger(__name__)


class OrchestrationError(Exception):
    """Exception raised when orchestration fails."""
    pass


@dataclass
class DocumentMetadata:
    """Metadata for processed document."""
    document_id: str
    title: Optional[str]
    authors: List[str]
    abstract: Optional[str]
    keywords: List[str]
    source: str
    processing_timestamp: str


@dataclass
class ExtractionPipeline:
    """Configuration for extraction pipeline."""
    extract_equations: bool = True
    analyze_dependencies: bool = True
    classify_sdes: bool = True
    validate_fidelity: bool = True
    create_models: bool = True
    min_confidence: float = 0.5
    max_equations: Optional[int] = None


@dataclass
class ProcessingStatistics:
    """Statistics from extraction processing."""
    total_processing_time: float
    equations_extracted: int
    variables_identified: int
    sde_models_created: int
    validation_errors: int
    fidelity_score: float
    success_rate: float


@dataclass
class OrchestrationResult:
    """Complete result of orchestrated extraction process."""
    document_metadata: DocumentMetadata
    equations: List[MathematicalEquation]
    variables: List[MathematicalVariable]
    dependency_graph: DependencyGraph
    sde_models: List[SDEModel]
    classification_results: List[ClassificationResult]
    statistics: ProcessingStatistics
    validation_report: Dict[str, Any]
    constitutional_compliance: bool


class ExtractionOrchestrator:
    """Master orchestrator for mathematical extraction pipeline.

    This class enforces constitutional mathematical fidelity requirements:
    - Coordinates all extraction components with fidelity preservation
    - Validates mathematical consistency across all processing stages
    - Ensures exact preservation of mathematical expressions throughout pipeline
    """

    def __init__(self,
                 latex_parser: Optional[LaTeXParser] = None,
                 equation_extractor: Optional[EquationExtractor] = None,
                 dependency_analyzer: Optional[DependencyAnalyzer] = None,
                 sde_classifier: Optional[SDEClassifier] = None):
        """Initialize extraction orchestrator.

        Args:
            latex_parser: LaTeX parser instance
            equation_extractor: Equation extractor instance
            dependency_analyzer: Dependency analyzer instance
            sde_classifier: SDE classifier instance
        """
        # Initialize components with constitutional fidelity
        self.latex_parser = latex_parser or LaTeXParser(strict_mode=True)
        self.equation_extractor = equation_extractor or EquationExtractor(self.latex_parser)
        self.dependency_analyzer = dependency_analyzer or DependencyAnalyzer(self.latex_parser)
        self.sde_classifier = sde_classifier or SDEClassifier(self.latex_parser, self.dependency_analyzer)

        # Initialize processing state
        self.processing_history = []
        self.validation_results = {}

    def process_document(self,
                        document_content: str,
                        document_id: str,
                        pipeline_config: Optional[ExtractionPipeline] = None,
                        metadata: Optional[DocumentMetadata] = None) -> OrchestrationResult:
        """Process a complete document through the extraction pipeline.

        Args:
            document_content: LaTeX document content
            document_id: Unique identifier for document
            pipeline_config: Configuration for extraction pipeline
            metadata: Optional document metadata

        Returns:
            OrchestrationResult: Complete extraction results

        Raises:
            OrchestrationError: If orchestration fails critically
        """
        start_time = time.time()
        pipeline_config = pipeline_config or ExtractionPipeline()

        try:
            logger.info(f"Starting orchestrated extraction for document {document_id}")

            # Step 1: Document preprocessing and metadata extraction
            processed_metadata = self._process_metadata(document_content, document_id, metadata)

            # Step 2: Extract equations
            equations = []
            extraction_result = None
            if pipeline_config.extract_equations:
                extraction_result = self._extract_equations(document_content, document_id, pipeline_config)
                equations = self._convert_to_mathematical_equations(extraction_result.extracted_equations)

            # Step 3: Analyze dependencies
            dependency_result = None
            dependency_graph = None
            variables = []
            if pipeline_config.analyze_dependencies and extraction_result:
                dependency_result = self._analyze_dependencies(extraction_result.extracted_equations)
                dependency_graph = dependency_result.dependency_graph
                variables = self._convert_to_mathematical_variables(dependency_result.variables)

            # Step 4: Classify SDE models
            classification_results = []
            sde_models = []
            if pipeline_config.classify_sdes and extraction_result:
                classification_results = self._classify_sde_models(extraction_result.extracted_equations, pipeline_config)
                if pipeline_config.create_models:
                    sde_models = self._create_sde_models(classification_results)

            # Step 5: Validate mathematical fidelity
            validation_report = {}
            constitutional_compliance = True
            if pipeline_config.validate_fidelity:
                validation_report, constitutional_compliance = self._validate_fidelity(
                    equations, variables, dependency_graph, sde_models, classification_results
                )

            # Step 6: Calculate statistics
            processing_time = time.time() - start_time
            statistics = self._calculate_statistics(
                processing_time, equations, variables, sde_models,
                validation_report, extraction_result
            )

            # Create final result
            result = OrchestrationResult(
                document_metadata=processed_metadata,
                equations=equations,
                variables=variables,
                dependency_graph=dependency_graph or DependencyGraph(nodes=[], edges=[], evaluation_order=[]),
                sde_models=sde_models,
                classification_results=classification_results,
                statistics=statistics,
                validation_report=validation_report,
                constitutional_compliance=constitutional_compliance
            )

            logger.info(f"Orchestrated extraction completed for document {document_id} in {processing_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Orchestration failed for document {document_id}: {str(e)}")
            raise OrchestrationError(f"Document processing failed: {str(e)}")

    def _process_metadata(self,
                         document_content: str,
                         document_id: str,
                         provided_metadata: Optional[DocumentMetadata]) -> DocumentMetadata:
        """Process and extract document metadata."""
        if provided_metadata:
            return provided_metadata

        # Extract metadata from LaTeX document
        title = self._extract_title(document_content)
        authors = self._extract_authors(document_content)
        abstract = self._extract_abstract(document_content)
        keywords = self._extract_keywords(document_content)

        return DocumentMetadata(
            document_id=document_id,
            title=title,
            authors=authors,
            abstract=abstract,
            keywords=keywords,
            source="latex_extraction",
            processing_timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )

    def _extract_title(self, document_content: str) -> Optional[str]:
        """Extract title from LaTeX document."""
        import re
        title_match = re.search(r'\\title\{([^}]+)\}', document_content)
        return title_match.group(1).strip() if title_match else None

    def _extract_authors(self, document_content: str) -> List[str]:
        """Extract authors from LaTeX document."""
        import re
        author_match = re.search(r'\\author\{([^}]+)\}', document_content)
        if author_match:
            authors_text = author_match.group(1)
            # Split by common separators
            authors = re.split(r'[,;]|\\and', authors_text)
            return [author.strip() for author in authors if author.strip()]
        return []

    def _extract_abstract(self, document_content: str) -> Optional[str]:
        """Extract abstract from LaTeX document."""
        import re
        abstract_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', document_content, re.DOTALL)
        return abstract_match.group(1).strip() if abstract_match else None

    def _extract_keywords(self, document_content: str) -> List[str]:
        """Extract keywords from LaTeX document."""
        import re
        # Look for common keyword patterns
        keywords_patterns = [
            r'\\keywords\{([^}]+)\}',
            r'Keywords:([^\n]+)',
            r'Key words:([^\n]+)'
        ]

        for pattern in keywords_patterns:
            match = re.search(pattern, document_content, re.IGNORECASE)
            if match:
                keywords_text = match.group(1)
                keywords = re.split(r'[,;]', keywords_text)
                return [kw.strip() for kw in keywords if kw.strip()]

        return []

    def _extract_equations(self,
                          document_content: str,
                          document_id: str,
                          pipeline_config: ExtractionPipeline) -> ExtractionResult:
        """Extract equations using the equation extractor."""
        try:
            result = self.equation_extractor.extract_equations(document_content, document_id)

            # Apply filtering based on pipeline configuration
            if pipeline_config.min_confidence > 0:
                result.extracted_equations = [
                    eq for eq in result.extracted_equations
                    if eq.confidence_score >= pipeline_config.min_confidence
                ]

            if pipeline_config.max_equations:
                result.extracted_equations = result.extracted_equations[:pipeline_config.max_equations]

            return result

        except Exception as e:
            logger.error(f"Equation extraction failed: {str(e)}")
            raise OrchestrationError(f"Equation extraction failed: {str(e)}")

    def _analyze_dependencies(self, extracted_equations: List[ExtractedEquation]) -> DependencyAnalysisResult:
        """Analyze dependencies using the dependency analyzer."""
        try:
            return self.dependency_analyzer.analyze_dependencies(extracted_equations)
        except Exception as e:
            logger.error(f"Dependency analysis failed: {str(e)}")
            raise OrchestrationError(f"Dependency analysis failed: {str(e)}")

    def _classify_sde_models(self,
                           extracted_equations: List[ExtractedEquation],
                           pipeline_config: ExtractionPipeline) -> List[ClassificationResult]:
        """Classify SDE models using the SDE classifier."""
        classification_results = []

        for equation in extracted_equations:
            if equation.equation_type.value == "SDE":
                try:
                    result = self.sde_classifier.classify_sde(equation)
                    if result.confidence_score >= pipeline_config.min_confidence:
                        classification_results.append(result)
                except Exception as e:
                    logger.warning(f"SDE classification failed for equation {equation.equation_id}: {str(e)}")

        return classification_results

    def _create_sde_models(self, classification_results: List[ClassificationResult]) -> List[SDEModel]:
        """Create SDE model instances from classification results."""
        sde_models = []

        for result in classification_results:
            try:
                sde_model = self.sde_classifier.create_sde_model(result)
                sde_models.append(sde_model)
            except Exception as e:
                logger.warning(f"SDE model creation failed for {result.equation_id}: {str(e)}")

        return sde_models

    def _convert_to_mathematical_equations(self, extracted_equations: List[ExtractedEquation]) -> List[MathematicalEquation]:
        """Convert extracted equations to MathematicalEquation instances."""
        equations = []

        for eq in extracted_equations:
            try:
                math_eq = MathematicalEquation(
                    equation_id=eq.equation_id,
                    latex_form=eq.latex_form,
                    equation_type=eq.equation_type.value,
                    variables=eq.variables,
                    parameters=eq.parameters,
                    source_location=eq.section_reference
                )
                equations.append(math_eq)
            except Exception as e:
                logger.warning(f"Failed to convert equation {eq.equation_id}: {str(e)}")

        return equations

    def _convert_to_mathematical_variables(self, variable_nodes: Dict[str, Any]) -> List[MathematicalVariable]:
        """Convert variable nodes to MathematicalVariable instances."""
        variables = []

        for symbol, node in variable_nodes.items():
            try:
                # Convert VariableRole to VariableType
                variable_type = convert_role_to_variable_type(node.role)

                math_var = MathematicalVariable(
                    symbol=symbol,
                    description=f"Mathematical variable {symbol}",  # Add required description
                    units=node.units if node.units else "dimensionless",  # Add required units
                    variable_type=variable_type,
                    domain=[float('-inf'), float('inf')],  # Default real domain as list
                    dependencies=node.dependencies if hasattr(node, 'dependencies') else []
                )
                variables.append(math_var)
            except Exception as e:
                logger.warning(f"Failed to convert variable {symbol}: {str(e)}")

        return variables

    def _validate_fidelity(self,
                          equations: List[MathematicalEquation],
                          variables: List[MathematicalVariable],
                          dependency_graph: Optional[DependencyGraph],
                          sde_models: List[SDEModel],
                          classification_results: List[ClassificationResult]) -> Tuple[Dict[str, Any], bool]:
        """Validate mathematical fidelity across all components."""
        validation_report = {
            "equations_validated": 0,
            "variables_validated": 0,
            "models_validated": 0,
            "fidelity_violations": [],
            "constitutional_compliance": True
        }

        constitutional_compliance = True

        # Validate equations
        for equation in equations:
            try:
                if equation.validate_constitutional_compliance():
                    validation_report["equations_validated"] += 1
                else:
                    validation_report["fidelity_violations"].append(f"Equation {equation.equation_id} failed fidelity check")
                    constitutional_compliance = False
            except Exception as e:
                validation_report["fidelity_violations"].append(f"Equation {equation.equation_id} validation error: {str(e)}")

        # Validate variables
        for variable in variables:
            try:
                if variable.validate_constitutional_compliance():
                    validation_report["variables_validated"] += 1
                else:
                    validation_report["fidelity_violations"].append(f"Variable {variable.symbol} failed fidelity check")
                    constitutional_compliance = False
            except Exception as e:
                validation_report["fidelity_violations"].append(f"Variable {variable.symbol} validation error: {str(e)}")

        # Validate SDE models
        for model in sde_models:
            try:
                if model.validate_mathematical_fidelity():
                    validation_report["models_validated"] += 1
                else:
                    validation_report["fidelity_violations"].append(f"SDE model {model.model_name} failed fidelity check")
                    constitutional_compliance = False
            except Exception as e:
                validation_report["fidelity_violations"].append(f"SDE model {model.model_name} validation error: {str(e)}")

        # Validate dependency graph
        if dependency_graph:
            try:
                if not dependency_graph.validate_constitutional_compliance():
                    validation_report["fidelity_violations"].append("Dependency graph failed constitutional compliance")
                    constitutional_compliance = False
            except Exception as e:
                validation_report["fidelity_violations"].append(f"Dependency graph validation error: {str(e)}")

        validation_report["constitutional_compliance"] = constitutional_compliance
        return validation_report, constitutional_compliance

    def _calculate_statistics(self,
                            processing_time: float,
                            equations: List[MathematicalEquation],
                            variables: List[MathematicalVariable],
                            sde_models: List[SDEModel],
                            validation_report: Dict[str, Any],
                            extraction_result: Optional[ExtractionResult]) -> ProcessingStatistics:
        """Calculate processing statistics."""
        equations_extracted = len(equations)
        variables_identified = len(variables)
        sde_models_created = len(sde_models)
        validation_errors = len(validation_report.get("fidelity_violations", []))

        # Calculate fidelity score
        total_validated = (validation_report.get("equations_validated", 0) +
                          validation_report.get("variables_validated", 0) +
                          validation_report.get("models_validated", 0))
        total_items = equations_extracted + variables_identified + sde_models_created
        fidelity_score = total_validated / max(total_items, 1)

        # Calculate success rate
        success_rate = extraction_result.success_rate if extraction_result else 1.0

        return ProcessingStatistics(
            total_processing_time=processing_time,
            equations_extracted=equations_extracted,
            variables_identified=variables_identified,
            sde_models_created=sde_models_created,
            validation_errors=validation_errors,
            fidelity_score=fidelity_score,
            success_rate=success_rate
        )

    def process_batch(self,
                     documents: List[Tuple[str, str]],
                     pipeline_config: Optional[ExtractionPipeline] = None) -> List[OrchestrationResult]:
        """Process multiple documents in batch.

        Args:
            documents: List of (content, document_id) tuples
            pipeline_config: Configuration for extraction pipeline

        Returns:
            List of OrchestrationResult for each document
        """
        results = []
        pipeline_config = pipeline_config or ExtractionPipeline()

        for content, doc_id in documents:
            try:
                result = self.process_document(content, doc_id, pipeline_config)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch processing failed for document {doc_id}: {str(e)}")
                # Continue processing other documents

        return results

    def validate_constitutional_compliance(self, result: OrchestrationResult) -> bool:
        """Validate constitutional mathematical fidelity requirements for orchestration result."""
        return result.constitutional_compliance