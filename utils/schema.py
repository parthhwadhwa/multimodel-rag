from pydantic import BaseModel

class DrugInfo(BaseModel):
    drug_name: str
    drug_class: str
    uses: str
    dosage_info: str
    common_side_effects: str
    serious_side_effects: str
    contraindications: str
    warnings: str
    source: str
