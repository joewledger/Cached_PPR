import Src.io_utils as io


def test_unique_proximity_filepaths():

    db_wrapper = io.DBWrapper("Cache/test.sqlite3")
    db_wrapper.open_connection()
    assert 10 == len(db_wrapper.get_unique_proximity_filepaths(count=10))
