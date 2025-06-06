"""add_fields_for_signature_analysis_and_proposal_ids

Revision ID: a0cff1a9e6ea
Revises: 5f92d650d1cc
Create Date: 2025-05-31 13:29:17.454600

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql # Added for JSON type


# revision identifiers, used by Alembic.
revision: str = 'a0cff1a9e6ea'
down_revision: Union[str, None] = '5f92d650d1cc'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('projects', sa.Column('application_number', sa.String(), nullable=True))
    op.add_column('projects', sa.Column('signature_analysis_report_html', sa.Text(), nullable=True))
    op.add_column('projects', sa.Column('signature_analysis_status', sa.String(), nullable=True, server_default='pending'))
    op.create_index(op.f('ix_projects_application_number'), 'projects', ['application_number'], unique=True)
    
    op.add_column('signature_instances', sa.Column('bounding_box_json', postgresql.JSON(astext_type=sa.Text()), nullable=True))
    op.add_column('signature_instances', sa.Column('textract_response_json', postgresql.JSON(astext_type=sa.Text()), nullable=True))
    op.add_column('signature_instances', sa.Column('cropped_signature_image', sa.LargeBinary(), nullable=True))
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('signature_instances', 'cropped_signature_image')
    op.drop_column('signature_instances', 'textract_response_json')
    op.drop_column('signature_instances', 'bounding_box_json')
    
    op.drop_index(op.f('ix_projects_application_number'), table_name='projects')
    op.drop_column('projects', 'signature_analysis_status')
    op.drop_column('projects', 'signature_analysis_report_html')
    op.drop_column('projects', 'application_number')
    # ### end Alembic commands ###
