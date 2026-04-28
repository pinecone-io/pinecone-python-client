Admin
=====

The ``Admin`` client manages organizations, projects, and API keys.  It uses OAuth2
client credentials (service account) rather than an API key, and is the right tool
for control-plane operations such as creating projects and rotating keys.

.. autoclass:: pinecone.admin.Admin
   :members:
   :undoc-members: False
   :show-inheritance:
   :special-members: __init__


Organizations Namespace
-----------------------

.. autoclass:: pinecone.admin.organizations.Organizations
   :members:
   :undoc-members: False
   :show-inheritance:


Projects Namespace
------------------

.. autoclass:: pinecone.admin.projects.Projects
   :members:
   :undoc-members: False
   :show-inheritance:


API Keys Namespace
------------------

.. autoclass:: pinecone.admin.api_keys.ApiKeys
   :members:
   :undoc-members: False
   :show-inheritance:
