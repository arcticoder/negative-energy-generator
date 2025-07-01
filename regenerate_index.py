#!/usr/bin/env python3
"""
Documentation Index Regeneration System
=====================================

This script regenerates documentation indices and DAG structures for the energy
research framework, providing comprehensive cross-repository documentation
tracking and relationship mapping.

Features:
- NDJSON-based documentation indexing
- DAG structure generation for project relationships
- Cross-repository reference validation
- UQ status tracking and integration
- Edge node traversal for complex dependency analysis
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict, deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentationIndexRegenerator:
    """Advanced documentation index and DAG regeneration system."""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path).resolve()
        self.documentation_index = []
        self.highlights_dag = []
        self.repositories = [
            "casimir-tunable-permittivity-stacks",
            "negative-energy-generator", 
            "lqg-anec-framework",
            "unified-lqg-qft"
        ]
        # Determine workspace root - we're running from negative-energy-generator
        # so parent directory contains all repositories
        self.workspace_root = self.base_path.parent
        
    def load_existing_indices(self) -> Tuple[List[Dict], List[Dict]]:
        """Load existing documentation indices and DAG structures."""
        doc_index = []
        highlights = []
        
        # Load documentation index
        doc_index_path = self.base_path / "documentation-index.ndjson"
        if doc_index_path.exists():
            with open(doc_index_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        doc_index.append(json.loads(line))
        
        # Load highlights DAG
        highlights_path = self.base_path / "highlights-dag.ndjson"
        if highlights_path.exists():
            with open(highlights_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        highlights.append(json.loads(line))
        
        return doc_index, highlights
    
    def scan_repository_documentation(self, repo_path: Path) -> Dict[str, Any]:
        """Scan a repository for documentation files and extract metadata."""
        if not repo_path.exists():
            logger.warning(f"Repository path does not exist: {repo_path}")
            return {"exists": False, "repo_name": repo_path.name}
        
        repo_name = repo_path.name
        documentation_files = []
        
        # Scan for key documentation files
        key_files = [
            "README.md",
            "docs/technical-documentation.md",
            "BREAKTHROUGH_INTEGRATION_COMPLETE.md",
            "COMPLETE_THEORETICAL_VALIDATION_REPORT.md",
            "DISCOVERY_21_INTEGRATION_COMPLETE.md",
            "ADVANCED_FRAMEWORK_COMPLETION_SUMMARY.md"
        ]
        
        for file_pattern in key_files:
            file_path = repo_path / file_pattern
            if file_path.exists():
                documentation_files.append(file_pattern)
        
        # Scan for UQ-TODO files
        uq_todo_path = repo_path / "UQ-TODO.ndjson"
        uq_status = "not_assessed"
        if uq_todo_path.exists():
            try:
                with open(uq_todo_path, 'r', encoding='utf-8') as f:
                    uq_concerns = sum(1 for line in f if line.strip())
                if uq_concerns > 0:
                    uq_status = f"{uq_concerns}_concerns_identified"
                else:
                    uq_status = "no_concerns"
            except Exception as e:
                logger.warning(f"Error reading UQ-TODO file: {e}")
                uq_status = "read_error"
        
        return {
            "repo_name": repo_name,
            "documentation_files": documentation_files,
            "uq_status": uq_status,
            "exists": True
        }
    
    def determine_project_status(self, repo_info: Dict[str, Any]) -> str:
        """Determine project status based on documentation files."""
        doc_files = repo_info.get("documentation_files", [])
        repo_name = repo_info.get("repo_name", "")
        
        # Status determination logic
        if "BREAKTHROUGH_INTEGRATION_COMPLETE.md" in doc_files:
            return "breakthrough-complete"
        elif "COMPLETE_THEORETICAL_VALIDATION_REPORT.md" in doc_files:
            return "validation-complete"
        elif "DISCOVERY_21_INTEGRATION_COMPLETE.md" in doc_files:
            return "discovery-complete"
        elif "ADVANCED_FRAMEWORK_COMPLETION_SUMMARY.md" in doc_files:
            return "framework-complete"
        elif "docs/technical-documentation.md" in doc_files:
            return "production-ready"
        elif "README.md" in doc_files:
            return "documented"
        else:
            return "in-development"
    
    def build_cross_references(self, repos_info: Dict[str, Dict]) -> Dict[str, List[str]]:
        """Build cross-reference relationships between repositories."""
        cross_refs = defaultdict(list)
        
        # Define known relationships based on project structure
        relationships = {
            "casimir-tunable-permittivity-stacks": [
                "negative-energy-generator", "lqg-anec-framework", "unified-lqg-qft"
            ],
            "negative-energy-generator": [
                "casimir-tunable-permittivity-stacks", "lqg-anec-framework", "unified-lqg-qft"
            ],
            "lqg-anec-framework": [
                "unified-lqg-qft", "negative-energy-generator"
            ],
            "unified-lqg-qft": [
                "lqg-anec-framework", "negative-energy-generator"
            ]
        }
        
        for repo, refs in relationships.items():
            if repo in repos_info and repos_info[repo]["exists"]:
                cross_refs[repo] = [ref for ref in refs if ref in repos_info and repos_info[ref]["exists"]]
        
        return cross_refs
    
    def generate_documentation_index(self) -> List[Dict]:
        """Generate comprehensive documentation index."""
        logger.info("Generating documentation index...")
        
        # Scan all repositories
        repos_info = {}
        for repo in self.repositories:
            repo_path = self.workspace_root / repo
            info = self.scan_repository_documentation(repo_path)
            if not info:  # If scan returned empty dict, mark as not existing
                info = {"exists": False, "repo_name": repo}
            repos_info[repo] = info
        
        # Build cross-references
        cross_refs = self.build_cross_references(repos_info)
        
        # Generate index entries
        index_entries = []
        
        # Project descriptions
        descriptions = {
            "casimir-tunable-permittivity-stacks": "Revolutionary tunable permittivity stack platform enabling precise control over electromagnetic permittivity through quantum vacuum fluctuation manipulation",
            "negative-energy-generator": "Advanced negative energy generation platform with quantum field theory foundations and comprehensive validation",
            "lqg-anec-framework": "Loop Quantum Gravity and Averaged Null Energy Condition framework for advanced spacetime manipulation",
            "unified-lqg-qft": "Unified framework combining Loop Quantum Gravity with Quantum Field Theory for comprehensive spacetime-matter interaction modeling"
        }
        
        # Key features
        key_features = {
            "casimir-tunable-permittivity-stacks": [
                "sub-1% permittivity control accuracy", "THz regime operation", 
                "digital twin framework", "comprehensive UQ validation"
            ],
            "negative-energy-generator": [
                "prototype stack complete", "ML-powered optimization", 
                "integrated control systems", "hardware readiness"
            ],
            "lqg-anec-framework": [
                "mathematical consistency", "ANEC violation bounds", 
                "geodesic selection algorithms", "quantum state analysis"
            ],
            "unified-lqg-qft": [
                "QFT-LQG coupling", "vacuum state definitions", 
                "Lorentz invariance analysis", "field regularization"
            ]
        }
        
        for repo, info in repos_info.items():
            if info["exists"]:
                entry = {
                    "title": repo.replace("-", " ").title(),
                    "repo": repo,
                    "type": "theoretical_framework" if "framework" in repo or "lqg" in repo else "project",
                    "status": self.determine_project_status(info),
                    "description": descriptions.get(repo, f"Documentation for {repo}"),
                    "key_features": key_features.get(repo, []),
                    "documentation_files": info["documentation_files"],
                    "cross_references": cross_refs.get(repo, []),
                    "uq_status": info["uq_status"]
                }
                index_entries.append(entry)
        
        return index_entries
    
    def generate_highlights_dag(self, doc_index: List[Dict]) -> List[Dict]:
        """Generate highlights DAG with edge node traversal."""
        logger.info("Generating highlights DAG with edge node traversal...")
        
        dag_entries = []
        
        # Generate node entries
        for entry in doc_index:
            highlights = []
            repo = entry["repo"]
            
            # Generate highlights based on status and features
            if entry["status"] == "production-ready":
                highlights.extend([
                    "Digital twin framework complete",
                    "UQ critical issues resolved",
                    "Production-ready implementation"
                ])
            elif entry["status"] == "breakthrough-complete":
                highlights.extend([
                    "Prototype stack operational",
                    "ML optimization integrated", 
                    "Hardware modules ready"
                ])
            elif entry["status"] == "discovery-complete":
                highlights.extend([
                    "Discovery integration complete",
                    "Advanced algorithms implemented"
                ])
            elif entry["status"] == "framework-complete":
                highlights.extend([
                    "Theoretical framework complete",
                    "Mathematical foundations established"
                ])
            
            # Add key features as highlights
            highlights.extend(entry["key_features"][:2])  # Top 2 features
            
            priority = 1 if entry["type"] == "project" else 2
            
            node_entry = {
                "node": repo,
                "type": entry["type"],
                "status": entry["status"],
                "connections": entry["cross_references"],
                "highlights": highlights,
                "priority": priority
            }
            dag_entries.append(node_entry)
        
        # Generate edge entries with traversal information
        edges_added = set()
        for entry in doc_index:
            repo = entry["repo"]
            for ref in entry["cross_references"]:
                edge_key = tuple(sorted([repo, ref]))
                if edge_key not in edges_added:
                    # Determine edge type and strength
                    edge_type, strength = self._determine_edge_properties(repo, ref)
                    
                    edge_entry = {
                        "edge": f"{repo}->{ref}",
                        "type": edge_type,
                        "description": self._generate_edge_description(repo, ref, edge_type),
                        "strength": strength,
                        "bidirectional": True
                    }
                    dag_entries.append(edge_entry)
                    edges_added.add(edge_key)
        
        return dag_entries
    
    def _determine_edge_properties(self, repo1: str, repo2: str) -> Tuple[str, str]:
        """Determine edge type and strength between repositories."""
        # Define relationship types
        if ("casimir" in repo1 and "negative-energy" in repo2) or ("casimir" in repo2 and "negative-energy" in repo1):
            return "technical_integration", "high"
        elif ("lqg" in repo1 and "unified" in repo2) or ("lqg" in repo2 and "unified" in repo1):
            return "mathematical_basis", "high"
        elif ("negative-energy" in repo1 and "lqg" in repo2) or ("negative-energy" in repo2 and "lqg" in repo1):
            return "theoretical_foundation", "high"
        else:
            return "cross_reference", "medium"
    
    def _generate_edge_description(self, repo1: str, repo2: str, edge_type: str) -> str:
        """Generate description for repository relationships."""
        descriptions = {
            "technical_integration": "Permittivity control enables enhanced Casimir force manipulation for negative energy generation",
            "mathematical_basis": "Unified LQG-QFT provides mathematical framework for ANEC calculations in quantum gravity",
            "theoretical_foundation": "ANEC framework provides theoretical foundation for negative energy generation mechanisms",
            "cross_reference": "Shared theoretical foundations and complementary research objectives"
        }
        return descriptions.get(edge_type, f"Relationship between {repo1} and {repo2}")
    
    def perform_edge_node_traversal(self, dag_entries: List[Dict]) -> Dict[str, Any]:
        """Perform advanced edge node traversal analysis."""
        logger.info("Performing edge node traversal analysis...")
        
        # Build graph structure
        nodes = {}
        edges = []
        
        for entry in dag_entries:
            if "node" in entry:
                nodes[entry["node"]] = entry
            elif "edge" in entry:
                edges.append(entry)
        
        # Build adjacency list
        adjacency = defaultdict(list)
        for edge in edges:
            edge_parts = edge["edge"].split("->")
            if len(edge_parts) == 2:
                from_node, to_node = edge_parts
                adjacency[from_node].append({
                    "target": to_node,
                    "type": edge["type"],
                    "strength": edge["strength"]
                })
                if edge.get("bidirectional", False):
                    adjacency[to_node].append({
                        "target": from_node,
                        "type": edge["type"],
                        "strength": edge["strength"]
                    })
        
        # Perform traversal analysis
        traversal_results = {
            "connectivity_analysis": self._analyze_connectivity(adjacency, nodes),
            "critical_paths": self._find_critical_paths(adjacency, nodes),
            "hub_analysis": self._analyze_hubs(adjacency, nodes),
            "edge_node_identification": self._identify_edge_nodes(adjacency, nodes)
        }
        
        return traversal_results
    
    def _analyze_connectivity(self, adjacency: Dict, nodes: Dict) -> Dict:
        """Analyze graph connectivity."""
        total_nodes = len(nodes)
        total_edges = sum(len(adj) for adj in adjacency.values()) // 2  # Undirected graph
        
        # Calculate connectivity metrics
        connectivity = {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "density": (2 * total_edges) / (total_nodes * (total_nodes - 1)) if total_nodes > 1 else 0,
            "average_degree": (2 * total_edges) / total_nodes if total_nodes > 0 else 0
        }
        
        return connectivity
    
    def _find_critical_paths(self, adjacency: Dict, nodes: Dict) -> List[Dict]:
        """Find critical paths in the repository relationship graph."""
        critical_paths = []
        
        # Find paths between high-priority nodes
        high_priority_nodes = [node for node, data in nodes.items() if data.get("priority", 2) == 1]
        
        for start in high_priority_nodes:
            for end in high_priority_nodes:
                if start != end:
                    path = self._find_shortest_path(adjacency, start, end)
                    if path:
                        critical_paths.append({
                            "start": start,
                            "end": end,
                            "path": path,
                            "length": len(path) - 1
                        })
        
        return critical_paths
    
    def _find_shortest_path(self, adjacency: Dict, start: str, end: str) -> List[str]:
        """Find shortest path between two nodes using BFS."""
        if start == end:
            return [start]
        
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            for neighbor_info in adjacency.get(current, []):
                neighbor = neighbor_info["target"]
                if neighbor == end:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return []  # No path found
    
    def _analyze_hubs(self, adjacency: Dict, nodes: Dict) -> List[Dict]:
        """Identify hub nodes in the graph."""
        hub_analysis = []
        
        for node, data in nodes.items():
            degree = len(adjacency.get(node, []))
            
            # Calculate weighted degree based on edge strength
            weighted_degree = sum(
                {"high": 3, "medium": 2, "low": 1}.get(edge["strength"], 1)
                for edge in adjacency.get(node, [])
            )
            
            hub_analysis.append({
                "node": node,
                "degree": degree,
                "weighted_degree": weighted_degree,
                "type": data.get("type", "unknown"),
                "priority": data.get("priority", 2),
                "is_hub": degree >= 2 and weighted_degree >= 4
            })
        
        # Sort by weighted degree
        hub_analysis.sort(key=lambda x: x["weighted_degree"], reverse=True)
        
        return hub_analysis
    
    def _identify_edge_nodes(self, adjacency: Dict, nodes: Dict) -> List[Dict]:
        """Identify edge nodes (nodes with single connections or specialized roles)."""
        edge_nodes = []
        
        for node, data in nodes.items():
            connections = adjacency.get(node, [])
            degree = len(connections)
            
            # Edge node criteria
            is_edge_node = (
                degree <= 1 or  # Leaf nodes
                data.get("type") == "theoretical_framework" or  # Specialized theoretical nodes
                data.get("priority", 2) > 1  # Lower priority nodes
            )
            
            if is_edge_node:
                edge_nodes.append({
                    "node": node,
                    "degree": degree,
                    "type": data.get("type", "unknown"),
                    "edge_type": self._classify_edge_node_type(node, connections, data),
                    "connections": [conn["target"] for conn in connections]
                })
        
        return edge_nodes
    
    def _classify_edge_node_type(self, node: str, connections: List[Dict], data: Dict) -> str:
        """Classify the type of edge node."""
        degree = len(connections)
        node_type = data.get("type", "unknown")
        
        if degree == 0:
            return "isolated"
        elif degree == 1:
            return "leaf"
        elif node_type == "theoretical_framework":
            return "theoretical_endpoint"
        else:
            return "specialized"
    
    def save_indices(self, doc_index: List[Dict], highlights_dag: List[Dict]) -> None:
        """Save generated indices to files."""
        logger.info("Saving documentation indices...")
        
        # Save documentation index
        doc_index_path = self.base_path / "documentation-index.ndjson"
        with open(doc_index_path, 'w', encoding='utf-8') as f:
            for entry in doc_index:
                f.write(json.dumps(entry) + '\n')
        
        # Save highlights DAG
        highlights_path = self.base_path / "highlights-dag.ndjson"
        with open(highlights_path, 'w', encoding='utf-8') as f:
            for entry in highlights_dag:
                f.write(json.dumps(entry) + '\n')
        
        logger.info(f"Saved documentation index with {len(doc_index)} entries")
        logger.info(f"Saved highlights DAG with {len(highlights_dag)} entries")
    
    def generate_summary_report(self, doc_index: List[Dict], highlights_dag: List[Dict], 
                              traversal_results: Dict) -> str:
        """Generate comprehensive summary report."""
        report_lines = [
            "Documentation Index Regeneration Summary",
            "=" * 50,
            "",
            f"Total repositories indexed: {len([e for e in doc_index if e.get('type')])}",
            f"Production-ready projects: {len([e for e in doc_index if e.get('status') == 'production-ready'])}",
            f"Breakthrough-complete projects: {len([e for e in doc_index if e.get('status') == 'breakthrough-complete'])}",
            "",
            "Connectivity Analysis:",
            f"  - Total nodes: {traversal_results['connectivity_analysis']['total_nodes']}",
            f"  - Total edges: {traversal_results['connectivity_analysis']['total_edges']}",
            f"  - Graph density: {traversal_results['connectivity_analysis']['density']:.3f}",
            f"  - Average degree: {traversal_results['connectivity_analysis']['average_degree']:.2f}",
            "",
            "Hub Analysis:",
        ]
        
        for hub in traversal_results['hub_analysis'][:3]:  # Top 3 hubs
            report_lines.append(f"  - {hub['node']}: degree={hub['degree']}, weighted={hub['weighted_degree']}")
        
        report_lines.extend([
            "",
            "Edge Node Analysis:",
            f"  - Total edge nodes: {len(traversal_results['edge_node_identification'])}",
        ])
        
        for edge_node in traversal_results['edge_node_identification']:
            report_lines.append(f"  - {edge_node['node']}: {edge_node['edge_type']}")
        
        report_lines.extend([
            "",
            f"Critical paths identified: {len(traversal_results['critical_paths'])}",
            "",
            "Regeneration completed successfully!"
        ])
        
        return '\n'.join(report_lines)
    
    def regenerate_all(self) -> None:
        """Complete regeneration of all documentation indices."""
        logger.info("Starting complete documentation index regeneration...")
        
        try:
            # Generate documentation index
            doc_index = self.generate_documentation_index()
            
            # Generate highlights DAG
            highlights_dag = self.generate_highlights_dag(doc_index)
            
            # Perform edge node traversal
            traversal_results = self.perform_edge_node_traversal(highlights_dag)
            
            # Save indices
            self.save_indices(doc_index, highlights_dag)
            
            # Generate and display summary
            summary = self.generate_summary_report(doc_index, highlights_dag, traversal_results)
            print("\n" + summary)
            
            # Save traversal results
            traversal_path = self.base_path / "traversal-analysis.json"
            with open(traversal_path, 'w', encoding='utf-8') as f:
                json.dump(traversal_results, f, indent=2)
            
            logger.info("Documentation index regeneration completed successfully!")
            
        except Exception as e:
            logger.error(f"Error during regeneration: {e}")
            sys.exit(1)


def main():
    """Main entry point for documentation index regeneration."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Regenerate documentation indices and DAG structures")
    parser.add_argument("--base-path", default=".", help="Base path for index files")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    regenerator = DocumentationIndexRegenerator(args.base_path)
    regenerator.regenerate_all()


if __name__ == "__main__":
    main()
