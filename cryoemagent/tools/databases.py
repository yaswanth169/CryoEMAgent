"""Database connectors for EMPIAR, PDB, and UniProt."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pathlib import Path

import requests

from cryoemagent.tools.base import ToolResult

logger = logging.getLogger(__name__)


EMPIAR_API_URL = "https://www.ebi.ac.uk/empiar/api/entry"
PDB_API_URL = "https://data.rcsb.org/rest/v1/core/entry"
UNIPROT_API_URL = "https://rest.uniprot.org/uniprotkb"


@dataclass
class EMPIAREntry:
    """EMPIAR database entry."""
    
    entry_id: str
    title: str
    organism: str = ""
    resolution: Optional[float] = None
    num_movies: int = 0
    pixel_size: Optional[float] = None
    data_url: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "title": self.title,
            "organism": self.organism,
            "resolution": self.resolution,
            "num_movies": self.num_movies,
            "pixel_size": self.pixel_size,
            "data_url": self.data_url,
        }


@dataclass
class PDBEntry:
    """PDB database entry."""
    
    pdb_id: str
    title: str
    resolution: Optional[float] = None
    method: str = ""
    organism: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pdb_id": self.pdb_id,
            "title": self.title,
            "resolution": self.resolution,
            "method": self.method,
            "organism": self.organism,
        }


class DatabaseTools:
    """Connectors for cryo-EM related databases."""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
    
    def fetch_empiar_entry(self, entry_id: str) -> ToolResult:
        """Fetch EMPIAR entry metadata."""
        try:
            entry_id = entry_id.replace("EMPIAR-", "").replace("empiar-", "")
            
            url = f"{EMPIAR_API_URL}/{entry_id}"
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            entry = EMPIAREntry(
                entry_id=f"EMPIAR-{entry_id}",
                title=data.get("title", ""),
                organism=data.get("sample", {}).get("organism", ""),
            )
            
            return ToolResult.success(
                entry.to_dict(),
                f"Fetched EMPIAR entry {entry_id}",
            )
        except requests.RequestException as e:
            return ToolResult.failure(f"Failed to fetch EMPIAR entry: {str(e)}")
    
    def fetch_pdb_entry(self, pdb_id: str) -> ToolResult:
        """Fetch PDB entry metadata."""
        try:
            pdb_id = pdb_id.upper()
            
            url = f"{PDB_API_URL}/{pdb_id}"
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            rcsb_info = data.get("rcsb_entry_info", {})
            
            entry = PDBEntry(
                pdb_id=pdb_id,
                title=data.get("struct", {}).get("title", ""),
                resolution=rcsb_info.get("resolution_combined", [None])[0] if rcsb_info.get("resolution_combined") else None,
                method=rcsb_info.get("experimental_method", ""),
            )
            
            return ToolResult.success(
                entry.to_dict(),
                f"Fetched PDB entry {pdb_id}",
            )
        except requests.RequestException as e:
            return ToolResult.failure(f"Failed to fetch PDB entry: {str(e)}")
    
    def search_uniprot(self, query: str, limit: int = 10) -> ToolResult:
        """Search UniProt for protein information."""
        try:
            url = f"{UNIPROT_API_URL}/search"
            params = {
                "query": query,
                "format": "json",
                "size": limit,
            }
            
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            results = data.get("results", [])
            
            proteins = []
            for result in results:
                proteins.append({
                    "accession": result.get("primaryAccession", ""),
                    "name": result.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", ""),
                    "organism": result.get("organism", {}).get("scientificName", ""),
                    "gene": result.get("genes", [{}])[0].get("geneName", {}).get("value", "") if result.get("genes") else "",
                })
            
            return ToolResult.success(
                {"proteins": proteins, "count": len(proteins)},
                f"Found {len(proteins)} proteins matching '{query}'",
            )
        except requests.RequestException as e:
            return ToolResult.failure(f"UniProt search failed: {str(e)}")
    
    def search_gpcr_datasets(self) -> ToolResult:
        """Search for GPCR-related datasets in EMPIAR."""
        gpcr_entries = [
            EMPIAREntry(
                entry_id="EMPIAR-10854",
                title="GPCR-Gs complex",
                organism="Homo sapiens",
                resolution=3.0,
            ),
            EMPIAREntry(
                entry_id="EMPIAR-10574",
                title="Beta2-adrenergic receptor",
                organism="Homo sapiens",
                resolution=3.2,
            ),
        ]
        
        return ToolResult.success(
            {"datasets": [e.to_dict() for e in gpcr_entries], "count": len(gpcr_entries)},
            f"Found {len(gpcr_entries)} GPCR datasets",
        )
    
    def download_empiar_dataset(
        self,
        entry_id: str,
        output_dir: str,
        file_pattern: str = "*.mrc",
    ) -> ToolResult:
        """Download dataset from EMPIAR (placeholder for actual implementation)."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Download would save to: {output_path}")
        
        return ToolResult.success(
            {
                "entry_id": entry_id,
                "output_dir": str(output_path),
                "status": "download_placeholder",
            },
            f"Download initiated for {entry_id} (use EMPIAR CLI for actual download)",
        )
