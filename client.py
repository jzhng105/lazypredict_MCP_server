#!/usr/bin/env python3
"""
LazyPredict MCP Client

A comprehensive client for interacting with the LazyPredict MCP server.
Provides both programmatic API and command-line interface.
"""

import asyncio
import json
import sys
import argparse
import os
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import pandas as pd

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class LazyPredictMCPClient:
    """Client for LazyPredict MCP Server"""
    
    def __init__(self, server_path: Optional[str] = None):
        """
        Initialize the client
        
        Args:
            server_path: Path to the MCP server script
        """
        self.server_path = server_path or "lazypred_server.py"
        self.session: Optional[ClientSession] = None
        self.connected = False
        self.stdio_client = None
        self.read_stream = None
        self.write_stream = None
        
    async def connect(self):
        """Connect to the MCP server"""
        try:
            server_params = StdioServerParameters(
                command="python",
                args=[self.server_path],
            )
            
            # Start the server and create session
            self.stdio_client = stdio_client(server_params)
            self.read_stream, self.write_stream = await self.stdio_client.__aenter__()
            self.session = await ClientSession(self.read_stream, self.write_stream).__aenter__()
            
            # Initialize
            await self.session.initialize()
            self.connected = True
            print("Connected to LazyPredict MCP Server")

            # List available tools
            tools = await self.list_tools()
            print(f"Available tools: {len(tools)}")
            for tool in tools[:5]:  # Show first 5 tools
                print(f"- {tool['name']}: {tool.get('description', '')[:100]}...")
            if len(tools) > 5:
                print(f"... and {len(tools) - 5} more tools")
            
        except Exception as e:
            print(f"Failed to connect to server: {e}")
            raise

    async def disconnect(self):
        """Disconnect from the MCP server"""
        if self.session:
            try:
                await self.session.__aexit__(None, None, None)
                await self.stdio_client.__aexit__(None, None, None)
                self.connected = False
                print("Disconnected from server")
            except Exception as e:
                print(f"Warning: Error during disconnect: {e}")

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools"""
        if not self.connected:
            raise RuntimeError("Not connected to server. Call connect() first.")
        
        tools_response = await self.session.list_tools()
        tools = getattr(tools_response, "tools", tools_response)
        
        normalized_tools = []
        for tool in tools:
            name = getattr(tool, "name", None) or tool.get("name", "")
            description = getattr(tool, "description", None) or tool.get("description", "")
            inputSchema = getattr(tool, "inputSchema", None) or tool.get("inputSchema", {})
            normalized_tools.append({
                "name": name,
                "description": description,
                "inputSchema": inputSchema,
            })
        return normalized_tools

    async def list_resources(self) -> List[Dict[str, Any]]:
        """List available resources"""
        if not self.connected:
            raise RuntimeError("Not connected to server. Call connect() first.")
        
        resources_response = await self.session.list_resources()
        resources = getattr(resources_response, "resources", resources_response)
        
        normalized_resources = []
        for resource in resources:
            uri = getattr(resource, "uri", None) or resource.get("uri", "")
            name = getattr(resource, "name", None) or resource.get("name", "")
            description = getattr(resource, "description", None) or resource.get("description", "")
            normalized_resources.append({
                "uri": uri,
                "name": name,
                "description": description,
            })
        return normalized_resources

    async def read_resource(self, uri: str) -> str:
        """Read a resource"""
        if not self.connected:
            raise RuntimeError("Not connected to server. Call connect() first.")
        
        result = await self.session.read_resource(uri)
        return result.contents[0].text if result.contents else ""

    async def _call_tool(self, tool_name: str, arguments: Dict[str, Any] = None) -> Any:
        """Call a tool on the MCP server"""
        if not self.connected:
            raise RuntimeError("Not connected to server. Call connect() first.")
        
        try:
            result = await self.session.call_tool(tool_name, arguments or {})
            if result.isError:
                error_msg = result.content[0].text if result.content else 'Unknown error'
                raise RuntimeError(f"Tool error: {error_msg}")
            return result.content[0].text if result.content else ""
        except Exception as e:
            raise RuntimeError(f"Failed to call tool {tool_name}: {e}")

    # =================== DATASET MANAGEMENT ===================
    
    async def generate_sample_dataset(self, dataset_type: str, n_samples: int = 1000,
                                    n_features: int = 20, **kwargs) -> Dict[str, Any]:
        """Generate synthetic dataset for testing"""
        args = {
            "dataset_type": dataset_type,
            "n_samples": n_samples,
            "n_features": n_features,
            **kwargs
        }
        result = await self._call_tool("generate_sample_dataset", args)
        return json.loads(result)

    async def load_builtin_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Load built-in sklearn dataset"""
        args = {"dataset_name": dataset_name}
        result = await self._call_tool("load_builtin_dataset", args)
        return json.loads(result)

    async def load_custom_dataset(self, data: str, target_column: str, 
                                dataset_type: str = "auto") -> Dict[str, Any]:
        """Load custom dataset from CSV data"""
        args = {
            "data": data,
            "target_column": target_column,
            "dataset_type": dataset_type
        }
        result = await self._call_tool("load_custom_dataset", args)
        return json.loads(result)

    async def preprocess_data(self, dataset_id: str, test_size: float = 0.2,
                            scale_features: bool = True, encode_labels: bool = False,
                            preprocessing_steps: List[str] = None) -> Dict[str, Any]:
        """Preprocess data for ML pipeline"""
        args = {
            "dataset_id": dataset_id,
            "test_size": test_size,
            "scale_features": scale_features,
            "encode_labels": encode_labels
        }
        if preprocessing_steps:
            args["preprocessing_steps"] = preprocessing_steps
        result = await self._call_tool("preprocess_data", args)
        return json.loads(result)

    # =================== MODEL COMPARISON ===================

    async def run_lazy_classifier(self, dataset_id: str, classifiers: List[str] = None,
                                 custom_metric: str = None, sort_by: str = "Accuracy",
                                 predictions: bool = False, fold: int = 5) -> Dict[str, Any]:
        """Run comprehensive classification model comparison"""
        args = {
            "dataset_id": dataset_id,
            "sort_by": sort_by,
            "predictions": predictions,
            "fold": fold
        }
        if classifiers:
            args["classifiers"] = classifiers
        if custom_metric:
            args["custom_metric"] = custom_metric
        result = await self._call_tool("run_lazy_classifier", args)
        return json.loads(result)

    async def run_lazy_regressor(self, dataset_id: str, regressors: List[str] = None,
                               custom_metric: str = None, sort_by: str = "R-Squared",
                               predictions: bool = False, fold: int = 5) -> Dict[str, Any]:
        """Run comprehensive regression model comparison"""
        args = {
            "dataset_id": dataset_id,
            "sort_by": sort_by,
            "predictions": predictions,
            "fold": fold
        }
        if regressors:
            args["regressors"] = regressors
        if custom_metric:
            args["custom_metric"] = custom_metric
        result = await self._call_tool("run_lazy_regressor", args)
        return json.loads(result)

    async def run_lazy_clusterer(self, dataset_id: str, n_clusters: int = 3,
                                clusterers: List[str] = None) -> Dict[str, Any]:
        """Run comprehensive clustering model comparison"""
        args = {
            "dataset_id": dataset_id,
            "n_clusters": n_clusters
        }
        if clusterers:
            args["clusterers"] = clusterers
        result = await self._call_tool("run_lazy_clusterer", args)
        return json.loads(result)

    # =================== ANALYSIS AND RECOMMENDATIONS ===================

    async def analyze_classification_results(self, result_id: str, 
                                           analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """Analyze classification results in detail"""
        args = {
            "result_id": result_id,
            "analysis_type": analysis_type
        }
        result = await self._call_tool("analyze_classification_results", args)
        return json.loads(result)

    async def analyze_regression_results(self, result_id: str, 
                                       analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """Analyze regression results in detail"""
        args = {
            "result_id": result_id,
            "analysis_type": analysis_type
        }
        result = await self._call_tool("analyze_regression_results", args)
        return json.loads(result)

    async def analyze_clustering_results(self, result_id: str, 
                                       analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """Analyze clustering results in detail"""
        args = {
            "result_id": result_id,
            "analysis_type": analysis_type
        }
        result = await self._call_tool("analyze_clustering_results", args)
        return json.loads(result)

    async def compare_model_types(self, dataset_id: str, 
                                model_types: List[str] = None) -> Dict[str, Any]:
        """Compare different types of models on the same dataset"""
        args = {"dataset_id": dataset_id}
        if model_types:
            args["model_types"] = model_types
        result = await self._call_tool("compare_model_types", args)
        return json.loads(result)

    async def cross_validate_best_models(self, result_id: str, top_n: int = 5,
                                       cv_folds: int = 5) -> Dict[str, Any]:
        """Perform cross-validation on top-performing models"""
        args = {
            "result_id": result_id,
            "top_n": top_n,
            "cv_folds": cv_folds
        }
        result = await self._call_tool("cross_validate_best_models", args)
        return json.loads(result)

    async def generate_model_recommendation(self, result_id: str, 
                                          business_context: str = "general",
                                          priority: str = "accuracy") -> Dict[str, Any]:
        """Generate model recommendations based on results and context"""
        args = {
            "result_id": result_id,
            "business_context": business_context,
            "priority": priority
        }
        result = await self._call_tool("generate_model_recommendation", args)
        return json.loads(result)

    # =================== VISUALIZATION AND REPORTING ===================

    async def create_performance_visualization_data(self, result_id: str) -> Dict[str, Any]:
        """Create data for performance visualizations"""
        args = {"result_id": result_id}
        result = await self._call_tool("create_performance_visualization_data", args)
        return json.loads(result)

    async def generate_comprehensive_report(self, result_id: str,
                                          include_recommendations: bool = True,
                                          include_technical_details: bool = True) -> str:
        """Generate detailed analysis report"""
        args = {
            "result_id": result_id,
            "include_recommendations": include_recommendations,
            "include_technical_details": include_technical_details
        }
        result = await self._call_tool("generate_comprehensive_report", args)
        # This tool returns a markdown string, not JSON
        return result

    # =================== UTILITY METHODS ===================

    async def get_available_models(self, model_type: str = "all") -> Dict[str, Any]:
        """Get list of available models by type"""
        args = {"model_type": model_type}
        result = await self._call_tool("get_available_models", args)
        return json.loads(result)

    async def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        return json.loads(await self.read_resource("lazy://status"))

    async def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return json.loads(await self.read_resource("lazy://config"))

    async def get_cache_status(self) -> Dict[str, Any]:
        """Get cache status"""
        return json.loads(await self.read_resource("lazy://cache"))

    # =================== HIGH-LEVEL WORKFLOWS ===================

    async def quick_classification_workflow(self, data_source: Union[str, Dict], 
                                          target_column: str = None,
                                          export_path: str = None) -> Dict[str, Any]:
        """
        Complete classification workflow from data to results
        
        Args:
            data_source: Either CSV data string, file path, or dataset specification dict
            target_column: Target column name (required for CSV)
            export_path: Optional path to save report
        """
        workflow_results = {}
        
        # Step 1: Load data
        if isinstance(data_source, str):
            if target_column is None:
                raise ValueError("target_column required for CSV data")
            dataset_result = await self.load_custom_dataset(data_source, target_column)
        elif isinstance(data_source, dict):
            if data_source.get("type") == "builtin":
                dataset_result = await self.load_builtin_dataset(data_source["name"])
            elif data_source.get("type") == "sample":
                dataset_result = await self.generate_sample_dataset(
                    "classification",
                    data_source.get("n_samples", 1000),
                    data_source.get("n_features", 20),
                    n_classes=data_source.get("n_classes", 2)
                )
            else:
                raise ValueError("Invalid data_source specification")
        else:
            raise ValueError("data_source must be string or dict")
        
        if "error" in dataset_result:
            return dataset_result
        
        workflow_results["dataset"] = dataset_result
        dataset_id = dataset_result["dataset_id"]
        
        # Step 2: Preprocess data
        preprocess_result = await self.preprocess_data(dataset_id, scale_features=True)
        if "error" in preprocess_result:
            return preprocess_result
        
        workflow_results["preprocessing"] = preprocess_result
        preprocessed_id = preprocess_result["preprocessed_dataset_id"]
        
        # Step 3: Run classification
        classification_result = await self.run_lazy_classifier(preprocessed_id, predictions=True)
        if "error" in classification_result:
            return classification_result
        
        workflow_results["classification"] = classification_result
        result_id = classification_result["result_id"]
        
        # Step 4: Analyze results
        analysis_result = await self.analyze_classification_results(result_id)
        workflow_results["analysis"] = analysis_result
        
        # Step 5: Get recommendations
        recommendation_result = await self.generate_model_recommendation(result_id)
        workflow_results["recommendations"] = recommendation_result
        
        # Step 6: Generate report
        report = await self.generate_comprehensive_report(result_id)
        workflow_results["report"] = report
        
        # Step 7: Save report if requested
        if export_path:
            try:
                with open(export_path, 'w') as f:
                    f.write(report)
                workflow_results["export_path"] = export_path
            except Exception as e:
                workflow_results["export_error"] = str(e)
        
        return workflow_results

    async def quick_regression_workflow(self, data_source: Union[str, Dict], 
                                      target_column: str = None,
                                      export_path: str = None) -> Dict[str, Any]:
        """
        Complete regression workflow from data to results
        
        Args:
            data_source: Either CSV data string, file path, or dataset specification dict
            target_column: Target column name (required for CSV)
            export_path: Optional path to save report
        """
        workflow_results = {}
        
        # Step 1: Load data
        if isinstance(data_source, str):
            if target_column is None:
                raise ValueError("target_column required for CSV data")
            dataset_result = await self.load_custom_dataset(data_source, target_column)
        elif isinstance(data_source, dict):
            if data_source.get("type") == "builtin":
                dataset_result = await self.load_builtin_dataset(data_source["name"])
            elif data_source.get("type") == "sample":
                dataset_result = await self.generate_sample_dataset(
                    "regression",
                    data_source.get("n_samples", 1000),
                    data_source.get("n_features", 20)
                )
            else:
                raise ValueError("Invalid data_source specification")
        else:
            raise ValueError("data_source must be string or dict")
        
        if "error" in dataset_result:
            return dataset_result
        
        workflow_results["dataset"] = dataset_result
        dataset_id = dataset_result["dataset_id"]
        
        # Step 2: Preprocess data
        preprocess_result = await self.preprocess_data(dataset_id, scale_features=True)
        if "error" in preprocess_result:
            return preprocess_result
        
        workflow_results["preprocessing"] = preprocess_result
        preprocessed_id = preprocess_result["preprocessed_dataset_id"]
        
        # Step 3: Run regression
        regression_result = await self.run_lazy_regressor(preprocessed_id, predictions=True)
        if "error" in regression_result:
            return regression_result
        
        workflow_results["regression"] = regression_result
        result_id = regression_result["result_id"]
        
        # Step 4: Analyze results
        analysis_result = await self.analyze_regression_results(result_id)
        workflow_results["analysis"] = analysis_result
        
        # Step 5: Get recommendations
        recommendation_result = await self.generate_model_recommendation(result_id)
        workflow_results["recommendations"] = recommendation_result
        
        # Step 6: Generate report
        report = await self.generate_comprehensive_report(result_id)
        workflow_results["report"] = report
        
        # Step 7: Save report if requested
        if export_path:
            try:
                with open(export_path, 'w') as f:
                    f.write(report)
                workflow_results["export_path"] = export_path
            except Exception as e:
                workflow_results["export_error"] = str(e)
        
        return workflow_results

    async def compare_all_models_workflow(self, data_source: Union[str, Dict], 
                                        target_column: str = None,
                                        include_clustering: bool = True,
                                        export_path: str = None) -> Dict[str, Any]:
        """
        Comprehensive model comparison across all available types
        
        Args:
            data_source: Data source specification
            target_column: Target column name (required for CSV)
            include_clustering: Whether to include clustering analysis
            export_path: Optional path to save combined report
        """
        workflow_results = {}
        
        # Step 1: Load and preprocess data (same as other workflows)
        if isinstance(data_source, str):
            if target_column is None:
                raise ValueError("target_column required for CSV data")
            dataset_result = await self.load_custom_dataset(data_source, target_column)
        elif isinstance(data_source, dict):
            if data_source.get("type") == "builtin":
                dataset_result = await self.load_builtin_dataset(data_source["name"])
            elif data_source.get("type") == "sample":
                # Auto-detect if not specified
                dataset_type = data_source.get("dataset_type", "classification")
                if dataset_type == "classification":
                    dataset_result = await self.generate_sample_dataset(
                        "classification",
                        data_source.get("n_samples", 1000),
                        data_source.get("n_features", 20),
                        n_classes=data_source.get("n_classes", 2)
                    )
                else:
                    dataset_result = await self.generate_sample_dataset(
                        "regression",
                        data_source.get("n_samples", 1000),
                        data_source.get("n_features", 20)
                    )
            else:
                raise ValueError("Invalid data_source specification")
        else:
            raise ValueError("data_source must be string or dict")
        
        if "error" in dataset_result:
            return dataset_result
        
        workflow_results["dataset"] = dataset_result
        dataset_id = dataset_result["dataset_id"]
        dataset_type = dataset_result["dataset_type"]
        
        # Step 2: Preprocess data
        preprocess_result = await self.preprocess_data(dataset_id, scale_features=True)
        if "error" in preprocess_result:
            return preprocess_result
        
        workflow_results["preprocessing"] = preprocess_result
        preprocessed_id = preprocess_result["preprocessed_dataset_id"]
        
        # Step 3: Compare all relevant model types
        model_types = []
        if dataset_type == "classification":
            model_types.append("classification")
        elif dataset_type == "regression":
            model_types.append("regression")
        
        if include_clustering:
            model_types.append("clustering")
        
        comparison_result = await self.compare_model_types(preprocessed_id, model_types)
        workflow_results["model_comparison"] = comparison_result
        
        # Step 4: Analyze each result type
        detailed_analysis = {}
        
        for model_type, results in comparison_result.get("comparison_results", {}).items():
            if "error" not in results:
                result_id = results["result_id"]
                
                if model_type == "classification":
                    analysis = await self.analyze_classification_results(result_id)
                    recommendations = await self.generate_model_recommendation(result_id)
                elif model_type == "regression":
                    analysis = await self.analyze_regression_results(result_id)
                    recommendations = await self.generate_model_recommendation(result_id)
                elif model_type == "clustering":
                    analysis = await self.analyze_clustering_results(result_id)
                    recommendations = {"clustering": "See analysis for clustering insights"}
                
                detailed_analysis[model_type] = {
                    "analysis": analysis,
                    "recommendations": recommendations
                }
        
        workflow_results["detailed_analysis"] = detailed_analysis
        
        # Step 5: Generate comprehensive report
        combined_reports = []
        for model_type, results in comparison_result.get("comparison_results", {}).items():
            if "error" not in results:
                result_id = results["result_id"]
                report = await self.generate_comprehensive_report(result_id)
                combined_reports.append(f"# {model_type.title()} Analysis\n\n{report}")
        
        combined_report = "\n\n" + "="*100 + "\n\n".join(combined_reports)
        workflow_results["combined_report"] = combined_report
        
        # Step 6: Save report if requested
        if export_path:
            try:
                with open(export_path, 'w') as f:
                    f.write(combined_report)
                workflow_results["export_path"] = export_path
            except Exception as e:
                workflow_results["export_error"] = str(e)
        
        return workflow_results


class LazyPredictCLI:
    """Command-line interface for LazyPredict MCP Client"""
    
    def __init__(self):
        self.client = LazyPredictMCPClient()
    
    async def run_cli(self, args):
        """Run CLI command"""
        await self.client.connect()
        
        try:
            if args.command == "status":
                result = await self.client.get_status()
                print(json.dumps(result, indent=2))
            
            elif args.command == "config":
                result = await self.client.get_config()
                print(json.dumps(result, indent=2))
            
            elif args.command == "tools":
                tools = await self.client.list_tools()
                print(f"Available tools ({len(tools)}):")
                for tool in tools:
                    print(f"\n{tool['name']}:")
                    print(f"  Description: {tool['description']}")
                    if 'inputSchema' in tool and 'properties' in tool['inputSchema']:
                        print("  Parameters:")
                        for param, details in tool['inputSchema']['properties'].items():
                            required = param in tool['inputSchema'].get('required', [])
                            req_str = " (required)" if required else " (optional)"
                            print(f"    - {param}{req_str}: {details.get('description', 'No description')}")
            
            elif args.command == "resources":
                resources = await self.client.list_resources()
                print(f"Available resources ({len(resources)}):")
                for resource in resources:
                    print(f"  {resource['uri']}: {resource['name']} - {resource['description']}")
            
            elif args.command == "models":
                result = await self.client.get_available_models(args.type)
                print(json.dumps(result, indent=2))
            
            elif args.command == "generate":
                result = await self.client.generate_sample_dataset(
                    args.type, args.samples, args.features,
                    n_classes=getattr(args, 'classes', 2)
                )
                print(f"Generated dataset: {result['dataset_id']}")
                print(f"Shape: {result['shape']}")
                print(f"Type: {result['dataset_type']}")
            
            elif args.command == "builtin":
                result = await self.client.load_builtin_dataset(args.name)
                print(f"Loaded dataset: {result['dataset_id']}")
                print(f"Shape: {result['shape']}")
                print(f"Description: {result['description'][:200]}...")
            
            elif args.command == "preprocess":
                result = await self.client.preprocess_data(
                    args.dataset_id, args.test_size, args.scale, args.encode
                )
                print(f"Preprocessed dataset: {result['preprocessed_dataset_id']}")
                print(f"Train shape: {result['data_splits']['X_train_shape']}")
                print(f"Test shape: {result['data_splits']['X_test_shape']}")
            
            elif args.command == "classify":
                result = await self.client.run_lazy_classifier(
                    args.dataset_id, 
                    classifiers=getattr(args, 'models', None),
                    predictions=args.predictions,
                    fold=args.folds
                )
                print(f"Classification complete: {result['result_id']}")
                print(f"Best model: {result['best_model']['name']} (Accuracy: {result['best_model']['accuracy']:.4f})")
                print(f"Total models tested: {result['summary_stats']['total_models']}")
            
            elif args.command == "regress":
                result = await self.client.run_lazy_regressor(
                    args.dataset_id,
                    regressors=getattr(args, 'models', None),
                    predictions=args.predictions,
                    fold=args.folds
                )
                print(f"Regression complete: {result['result_id']}")
                print(f"Best model: {result['best_model']['name']} (R²: {result['best_model']['r_squared']:.4f})")
                print(f"Total models tested: {result['summary_stats']['total_models']}")
            
            elif args.command == "cluster":
                result = await self.client.run_lazy_clusterer(
                    args.dataset_id, args.clusters,
                    clusterers=getattr(args, 'models', None)
                )
                print(f"Clustering complete: {result['result_id']}")
                print(f"Best model: {result['best_model']['name']} (Silhouette: {result['best_model']['silhouette_score']:.4f})")
                print(f"Total models tested: {result['summary_stats']['total_models']}")
            
            elif args.command == "analyze":
                if args.type == "classification":
                    result = await self.client.analyze_classification_results(args.result_id, args.analysis_type)
                elif args.type == "regression":
                    result = await self.client.analyze_regression_results(args.result_id, args.analysis_type)
                elif args.type == "clustering":
                    result = await self.client.analyze_clustering_results(args.result_id, args.analysis_type)
                print(json.dumps(result, indent=2))
            
            elif args.command == "recommend":
                result = await self.client.generate_model_recommendation(
                    args.result_id, args.context, args.priority
                )
                print(json.dumps(result, indent=2))
            
            elif args.command == "report":
                result = await self.client.generate_comprehensive_report(
                    args.result_id, args.recommendations, args.technical
                )
                if args.output:
                    with open(args.output, 'w') as f:
                        f.write(result)
                    print(f"Report saved to {args.output}")
                else:
                    print(result)
            
            elif args.command == "quick-classify":
                if args.sample:
                    data_spec = {
                        "type": "sample",
                        "dataset_type": "classification",
                        "n_samples": args.samples,
                        "n_features": args.features,
                        "n_classes": args.classes
                    }
                else:
                    if not args.file:
                        raise ValueError("Either --sample or --file must be specified")
                    with open(args.file, 'r') as f:
                        csv_data = f.read()
                    data_spec = csv_data
                
                result = await self.client.quick_classification_workflow(
                    data_spec, args.target, args.output
                )
                
                if "error" in result:
                    print(f"Error: {result['error']}")
                else:
                    print("Classification workflow completed!")
                    print(f"Best model: {result['classification']['best_model']['name']}")
                    print(f"Accuracy: {result['classification']['best_model']['accuracy']:.4f}")
                    if args.output:
                        print(f"Report saved to: {args.output}")
            
            elif args.command == "quick-regress":
                if args.sample:
                    data_spec = {
                        "type": "sample",
                        "dataset_type": "regression",
                        "n_samples": args.samples,
                        "n_features": args.features
                    }
                else:
                    if not args.file:
                        raise ValueError("Either --sample or --file must be specified")
                    with open(args.file, 'r') as f:
                        csv_data = f.read()
                    data_spec = csv_data
                
                result = await self.client.quick_regression_workflow(
                    data_spec, args.target, args.output
                )
                
                if "error" in result:
                    print(f"Error: {result['error']}")
                else:
                    print("Regression workflow completed!")
                    print(f"Best model: {result['regression']['best_model']['name']}")
                    print(f"R²: {result['regression']['best_model']['r_squared']:.4f}")
                    if args.output:
                        print(f"Report saved to: {args.output}")
            
            elif args.command == "compare-all":
                if args.sample:
                    data_spec = {
                        "type": "sample",
                        "dataset_type": args.task_type,
                        "n_samples": args.samples,
                        "n_features": args.features
                    }
                    if args.task_type == "classification":
                        data_spec["n_classes"] = args.classes
                else:
                    if not args.file:
                        raise ValueError("Either --sample or --file must be specified")
                    with open(args.file, 'r') as f:
                        csv_data = f.read()
                    data_spec = csv_data
                
                result = await self.client.compare_all_models_workflow(
                    data_spec, args.target, args.clustering, args.output
                )
                
                if "error" in result:
                    print(f"Error: {result['error']}")
                else:
                    print("Model comparison workflow completed!")
                    print(f"Dataset type: {result['dataset']['dataset_type']}")
                    print("Models compared:")
                    for model_type in result.get("model_comparison", {}).get("models_compared", []):
                        print(f"  - {model_type}")
                    if args.output:
                        print(f"Combined report saved to: {args.output}")
            
            else:
                print("Unknown command")
            
        finally:
            await self.client.disconnect()


def create_parser():
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(description="LazyPredict MCP Client")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # System commands
    subparsers.add_parser("status", help="Check system status")
    subparsers.add_parser("config", help="Show configuration")
    subparsers.add_parser("tools", help="List available tools")
    subparsers.add_parser("resources", help="List available resources")
    
    # Models command
    models_parser = subparsers.add_parser("models", help="List available models")
    models_parser.add_argument("--type", choices=["classification", "regression", "clustering", "all"], 
                              default="all", help="Model type to list")
    
    # Data generation
    generate_parser = subparsers.add_parser("generate", help="Generate sample dataset")
    generate_parser.add_argument("type", choices=["classification", "regression", "clustering"])
    generate_parser.add_argument("--samples", type=int, default=1000, help="Number of samples")
    generate_parser.add_argument("--features", type=int, default=20, help="Number of features")
    generate_parser.add_argument("--classes", type=int, default=2, help="Number of classes (classification only)")
    
    # Built-in datasets
    builtin_parser = subparsers.add_parser("builtin", help="Load built-in dataset")
    builtin_parser.add_argument("name", choices=["iris", "diabetes", "wine", "breast_cancer"])
    
    # Preprocessing
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess dataset")
    preprocess_parser.add_argument("dataset_id", help="Dataset ID to preprocess")
    preprocess_parser.add_argument("--test-size", type=float, default=0.2, help="Test set size")
    preprocess_parser.add_argument("--no-scale", dest="scale", action="store_false", help="Don't scale features")
    preprocess_parser.add_argument("--encode", action="store_true", help="Encode labels")
    
    # Classification
    classify_parser = subparsers.add_parser("classify", help="Run classification")
    classify_parser.add_argument("dataset_id", help="Preprocessed dataset ID")
    classify_parser.add_argument("--models", nargs="+", help="Specific models to run")
    classify_parser.add_argument("--predictions", action="store_true", help="Return predictions")
    classify_parser.add_argument("--folds", type=int, default=5, help="CV folds")
    
    # Regression
    regress_parser = subparsers.add_parser("regress", help="Run regression")
    regress_parser.add_argument("dataset_id", help="Preprocessed dataset ID")
    regress_parser.add_argument("--models", nargs="+", help="Specific models to run")
    regress_parser.add_argument("--predictions", action="store_true", help="Return predictions")
    regress_parser.add_argument("--folds", type=int, default=5, help="CV folds")
    
    # Clustering
    cluster_parser = subparsers.add_parser("cluster", help="Run clustering")
    cluster_parser.add_argument("dataset_id", help="Dataset ID")
    cluster_parser.add_argument("--clusters", type=int, default=3, help="Number of clusters")
    cluster_parser.add_argument("--models", nargs="+", help="Specific models to run")
    
    # Analysis
    analyze_parser = subparsers.add_parser("analyze", help="Analyze results")
    analyze_parser.add_argument("result_id", help="Result ID to analyze")
    analyze_parser.add_argument("type", choices=["classification", "regression", "clustering"])
    analyze_parser.add_argument("--analysis-type", choices=["comprehensive", "model_ranking", "performance_distribution", "model_families", "efficiency_analysis", "error_analysis", "clustering_quality"], default="comprehensive")
    
    # Recommendations
    recommend_parser = subparsers.add_parser("recommend", help="Generate recommendations")
    recommend_parser.add_argument("result_id", help="Result ID")
    recommend_parser.add_argument("--context", choices=["general", "financial", "healthcare", "marketing", "manufacturing"], default="general")
    recommend_parser.add_argument("--priority", choices=["accuracy", "speed", "balanced"], default="accuracy")
    
    # Report generation
    report_parser = subparsers.add_parser("report", help="Generate comprehensive report")
    report_parser.add_argument("result_id", help="Result ID")
    report_parser.add_argument("--output", help="Output file path")
    report_parser.add_argument("--no-recommendations", dest="recommendations", action="store_false")
    report_parser.add_argument("--no-technical", dest="technical", action="store_false")
    
    # Quick workflows
    quick_classify_parser = subparsers.add_parser("quick-classify", help="Quick classification workflow")
    quick_classify_group = quick_classify_parser.add_mutually_exclusive_group(required=True)
    quick_classify_group.add_argument("--file", help="CSV file path")
    quick_classify_group.add_argument("--sample", action="store_true", help="Use sample data")
    quick_classify_parser.add_argument("--target", help="Target column (required for CSV)")
    quick_classify_parser.add_argument("--samples", type=int, default=1000, help="Number of samples")
    quick_classify_parser.add_argument("--features", type=int, default=20, help="Number of features")
    quick_classify_parser.add_argument("--classes", type=int, default=2, help="Number of classes")
    quick_classify_parser.add_argument("--output", help="Report output file")
    
    quick_regress_parser = subparsers.add_parser("quick-regress", help="Quick regression workflow")
    quick_regress_group = quick_regress_parser.add_mutually_exclusive_group(required=True)
    quick_regress_group.add_argument("--file", help="CSV file path")
    quick_regress_group.add_argument("--sample", action="store_true", help="Use sample data")
    quick_regress_parser.add_argument("--target", help="Target column (required for CSV)")
    quick_regress_parser.add_argument("--samples", type=int, default=1000, help="Number of samples")
    quick_regress_parser.add_argument("--features", type=int, default=20, help="Number of features")
    quick_regress_parser.add_argument("--output", help="Report output file")
    
    # Compare all models
    compare_all_parser = subparsers.add_parser("compare-all", help="Compare all model types")
    compare_all_group = compare_all_parser.add_mutually_exclusive_group(required=True)
    compare_all_group.add_argument("--file", help="CSV file path")
    compare_all_group.add_argument("--sample", action="store_true", help="Use sample data")
    compare_all_parser.add_argument("--target", help="Target column (required for CSV)")
    compare_all_parser.add_argument("--task-type", choices=["classification", "regression"], default="classification")
    compare_all_parser.add_argument("--samples", type=int, default=1000, help="Number of samples")
    compare_all_parser.add_argument("--features", type=int, default=20, help="Number of features")
    compare_all_parser.add_argument("--classes", type=int, default=2, help="Number of classes")
    compare_all_parser.add_argument("--no-clustering", dest="clustering", action="store_false", help="Skip clustering analysis")
    compare_all_parser.add_argument("--output", help="Combined report output file")
    
    return parser


async def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = LazyPredictCLI()
    await cli.run_cli(args)


# Example usage as a library
async def example_usage():
    """Example of using the client as a library"""
    client = LazyPredictMCPClient()
    
    try:
        await client.connect()
        
        # Check system status
        # status = await client.get_status()
        # print("System Status:")
        # print(json.dumps(status, indent=2))
        
        # Example 1: Quick classification with built-in dataset
        print("\n" + "="*60)
        print("Example 1: Quick Classification with Iris Dataset")
        print("="*60)
        
        result = await client.quick_classification_workflow(
            {"type": "builtin", "name": "iris"},
            export_path="iris_classification_report.md"
        )
        
        if "error" not in result:
            print(f"Best Model: {result['classification']['best_model']['name']}")
            print(f"Accuracy: {result['classification']['best_model']['accuracy']:.4f}")
            print("Report saved to: iris_classification_report.md")
        
        # Example 2: Quick regression with built-in dataset
        print("\n" + "="*60)
        print("Example 2: Quick Regression with Diabetes Dataset")
        print("="*60)
        
        result = await client.quick_regression_workflow(
            {"type": "builtin", "name": "diabetes"},
            export_path="diabetes_regression_report.md"
        )
        
        if "error" not in result:
            print(f"Best Model: {result['regression']['best_model']['name']}")
            print(f"R²: {result['regression']['best_model']['r_squared']:.4f}")
            print("Report saved to: diabetes_regression_report.md")
        
        # Example 3: Generate synthetic data and compare all models
        print("\n" + "="*60)
        print("Example 3: Synthetic Data with Complete Model Comparison")
        print("="*60)
        
        result = await client.compare_all_models_workflow(
            {
                "type": "sample",
                "dataset_type": "classification",
                "n_samples": 1000,
                "n_features": 15,
                "n_classes": 3
            },
            include_clustering=True,
            export_path="complete_model_comparison.md"
        )
        
        if "error" not in result:
            print(f"Dataset Type: {result['dataset']['dataset_type']}")
            print(f"Models Compared: {result['model_comparison']['models_compared']}")
            print("Complete report saved to: complete_model_comparison.md")
        
        # Example 4: Advanced workflow with custom analysis
        print("\n" + "="*60)
        print("Example 4: Advanced Analysis Workflow")
        print("="*60)
        
        # Generate sample data
        dataset = await client.generate_sample_dataset("classification", 500, 10, n_classes=2)
        dataset_id = dataset["dataset_id"]
        
        # Preprocess
        preprocessed = await client.preprocess_data(dataset_id, scale_features=True)
        preprocessed_id = preprocessed["preprocessed_dataset_id"]
        
        # Run classification
        classification = await client.run_lazy_classifier(preprocessed_id, predictions=True)
        result_id = classification["result_id"]
        
        # Detailed analysis
        analysis = await client.analyze_classification_results(result_id, "model_families")
        print("Model Family Analysis:")
        if "model_families" in analysis:
            for family, info in analysis["model_families"].items():
                print(f"  {family}: {info['count']} models, avg accuracy: {info['mean_accuracy']:.4f}")
        
        # Cross-validation
        cv_results = await client.cross_validate_best_models(result_id, top_n=3)
        print("Cross-Validation Results:")
        for model, cv_data in cv_results["cv_results"].items():
            print(f"  {model}: {cv_data['cv_mean']:.4f} ± {cv_data['cv_std']:.4f}")
        
        # Recommendations
        recommendations = await client.generate_model_recommendation(result_id, "healthcare", "balanced")
        print(f"Recommended Model: {recommendations['primary_recommendation']['model_name']}")
        print(f"Selection Criteria: {recommendations['primary_recommendation']['selection_criteria']}")
        
        # Visualization data
        viz_data = await client.create_performance_visualization_data(result_id)
        print(f"Visualization data prepared for {len(viz_data['model_names'])} models")
        
        print("\nAdvanced analysis completed!")
        
    except Exception as e:
        print(f"Error in example: {e}")
        
    finally:
        await client.disconnect()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run CLI
        asyncio.run(main())
    else:
        # Run example
        print("Running LazyPredict MCP Client Examples...")
        print("Use 'python client.py --help' to see CLI options")
        print("\nRunning example workflows...")
        asyncio.run(example_usage())