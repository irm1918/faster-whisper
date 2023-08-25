import os

from faster_whisper import download_model


def test_download_model(tmpdir):
    """
    This function tests the 'download_model' function from the 'faster_whisper' module. It checks if the 
    downloaded model is correctly saved in the specified output directory and verifies that the directory 
    is not a symbolic link. It also checks that all files in the model directory are not symbolic links.
    """
    output_dir = str(tmpdir.join("model"))

    model_dir = download_model("tiny", output_dir=output_dir)

    assert model_dir == output_dir
    assert os.path.isdir(model_dir)
    assert not os.path.islink(model_dir)

    for filename in os.listdir(model_dir):
        path = os.path.join(model_dir, filename)
        assert not os.path.islink(path)


def test_download_model_in_cache(tmpdir):
    """
    This function tests the 'download_model' function with a cache directory. It checks if the model is 
    correctly downloaded and saved in the cache directory.
    """
    cache_dir = str(tmpdir.join("model"))
    download_model("tiny", cache_dir=cache_dir)
    assert os.path.isdir(cache_dir)
