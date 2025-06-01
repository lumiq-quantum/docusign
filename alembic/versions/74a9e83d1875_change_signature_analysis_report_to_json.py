"""change_signature_analysis_report_to_json

Revision ID: 74a9e83d1875
Revises: 9adc6d6ff546
Create Date: 2025-06-01 05:27:55.358261

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '74a9e83d1875'
down_revision: Union[str, None] = '9adc6d6ff546'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # To prevent errors from existing non-JSON data (e.g., HTML),
    # we set the column to NULL before changing its type.
    # This means any existing data in this column will be lost.
    # op.execute("UPDATE projects SET signature_analysis_report_html = NULL WHERE signature_analysis_report_html IS NOT NULL")

    # op.alter_column('projects',
    #                 'signature_analysis_report_html',
    #                 new_column_name='signature_analysis_report_json',
    #                 type_=sa.JSON(),
    #                 existing_type=sa.Text(),
    #                 postgresql_using='signature_analysis_report_html::json')


def downgrade() -> None:
    """Downgrade schema."""
    # op.alter_column('projects',
    #                 'signature_analysis_report_json',
    #                 new_column_name='signature_analysis_report_html',
    #                 type_=sa.Text(),
    #                 existing_type=sa.JSON(),
    #                 postgresql_using='signature_analysis_report_json::text')
