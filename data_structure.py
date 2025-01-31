# Define structured data model
class TransformLog(BaseModel):
    needs_review: bool
    lost_data: List[str]
    modified_data: List[str]
    other_data_modifications: List[str]
    comment: str

class Product(BaseModel):
    name: Optional[str]
    description: Optional[str]
    location: List[int]
    category: Optional[str]
    id: str
    is_active: bool

class ProductScheme(BaseModel):
    product: Product
    log: TransformLog

class ProductList(BaseModel):
    items: List[ProductScheme]
