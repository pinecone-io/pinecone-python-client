Preview
=======

.. admonition:: Preview — not covered by SemVer
   :class: warning

   The ``preview`` namespace exposes pre-release API features. Signatures,
   behavior, and availability may change in any minor SDK release without
   notice. **Pin your SDK version** when relying on preview features.
   Preview features are not subject to the SDK's normal deprecation policy.

Access via ``pc.preview`` on a :class:`~pinecone.Pinecone` instance, or
``pc.preview`` on an :class:`~pinecone.AsyncPinecone` instance.

Preview (sync)
--------------

.. autoclass:: pinecone.preview.Preview
   :members:
   :undoc-members: False
   :show-inheritance:
   :special-members: __init__

PreviewIndexes
--------------

.. autoclass:: pinecone.preview.indexes.PreviewIndexes
   :members:
   :undoc-members: False
   :show-inheritance:

PreviewIndex
------------

.. autoclass:: pinecone.preview.index.PreviewIndex
   :members:
   :undoc-members: False
   :show-inheritance:

Preview (async)
---------------

.. autoclass:: pinecone.preview.AsyncPreview
   :members:
   :undoc-members: False
   :show-inheritance:
   :special-members: __init__

AsyncPreviewIndexes
-------------------

.. autoclass:: pinecone.preview.async_indexes.AsyncPreviewIndexes
   :members:
   :undoc-members: False
   :show-inheritance:

AsyncPreviewIndex
-----------------

.. autoclass:: pinecone.preview.async_index.AsyncPreviewIndex
   :members:
   :undoc-members: False
   :show-inheritance:
