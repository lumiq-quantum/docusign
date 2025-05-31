"""add_chat_session_id_to_documents

Revision ID: 9adc6d6ff546
Revises: a0cff1a9e6ea
Create Date: 2025-06-01 03:34:34.063253

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9adc6d6ff546'
down_revision: Union[str, None] = 'a0cff1a9e6ea'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('documents', sa.Column('chat_session_id', sa.String(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('documents', 'chat_session_id')
