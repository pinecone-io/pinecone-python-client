from pinecone import Vector, Config


class TestDatetimeConversion:
    def test_datetimes_not_coerced(self):
        vec = Vector(
            id="1",
            values=[0.1, 0.2, 0.3],
            metadata={"created_at": "7th of January, 2023"},
            _check_type=True,
            _configuration=Config(),
        )
        assert vec.metadata["created_at"] == "7th of January, 2023"
        assert isinstance(vec.metadata["created_at"], str)

    def test_dates_not_coerced(self):
        vec = Vector(
            id="1",
            values=[0.1, 0.2, 0.3],
            metadata={"created_at": "8/12/2024"},
            _check_type=True,
            _configuration=Config(),
        )
        assert vec.metadata["created_at"] == "8/12/2024"
        assert isinstance(vec.metadata["created_at"], str)
