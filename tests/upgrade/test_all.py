class TestAll:
    def test_all_is_complete(self):
        """Test that __all__ is complete and accurate."""
        # Import the module
        import pinecone

        # Get all public names (those that don't start with _)
        public_names = {name for name in dir(pinecone) if not name.startswith("_")}

        # Get __all__ if it exists, otherwise empty set
        all_names = set(getattr(pinecone, "__all__", []))

        # Check that __all__ exists
        assert hasattr(pinecone, "__all__"), "Module should have __all__ defined"

        # Check that all names in __all__ are actually importable
        for name in all_names:
            assert getattr(pinecone, name) is not None, f"Name {name} in __all__ is not importable"

        # Check that all public names are in __all__
        missing_from_all = public_names - all_names
        for name in missing_from_all:
            print(f"Public name {name} is not in __all__")
        assert not missing_from_all, f"Public names not in __all__: {missing_from_all}"

        # Check that __all__ doesn't contain any private names
        private_in_all = {name for name in all_names if name.startswith("_")}
        assert not private_in_all, f"Private names in __all__: {private_in_all}"
