MusicDNA Detector
=================

MusicDNA Detector extracts pitch and note-duration events from short,
predominantly monophonic humming and singing queries. This is release ``0.9.0``,
a preliminary, pre-1.0 release; the documented
``musicdna-analysis-v0`` and ``musicdna-events-v0`` contracts may still change
before 1.0.

Installation
------------

Python 3.10 or newer is required. Either clone the repository (or download
and extract its archive) into any directory, then run the installation from
the repository root::

    git clone https://github.com/dh-erfurt/MusicDNA-Detector.git
    cd MusicDNA-Detector
    python -m pip install .

Or install directly from GitHub::

    python -m pip install "musicdna-detector @ git+https://github.com/dh-erfurt/MusicDNA-Detector.git"

See the `pip VCS installation documentation
<https://pip.pypa.io/en/stable/topics/vcs-support/>`_ for the installation
syntax. The optional librosa reference backend can be installed with::

    python -m pip install "musicdna-detector[librosa]"

Use
---

For a command-line workflow, the installed ``musicdna-detector`` command
analyzes one audio file and writes ``musicdna-analysis-v0`` JSON to stdout::

    musicdna-detector path/to/query.wav --profile humming

Use ``--output`` to write the JSON to a file for the MusicDNA-Encoder::

    musicdna-detector path/to/query.wav --profile humming --output detector-output.json

For Python code, pass the path of an audio file to ``analyze_file``. The file
can be anywhere; there is no required audio directory. Use a relative path
from the current directory or an absolute path. For humming, use the
``humming`` profile; for singing, use the ``singing`` profile.
If no profile is given, ``default`` is used; in version 0.9 it has the same
settings as ``singing``::

    from musicdna_detector import AnalysisConfig, analyze_file

    result = analyze_file(
        "path/to/query.wav",
        AnalysisConfig(profile="humming"),
    )
    print(result.to_json(indent=2))

The repository includes ``examples/synthetic_melody.wav``, a short synthetic
test signal that is useful for checking an installation. It is not a recording
of a singer or a humming performer. Try it with::

    from musicdna_detector import AnalysisConfig, analyze_file

    result = analyze_file(
        "examples/synthetic_melody.wav",
        AnalysisConfig(profile="humming"),
    )
    output = result.to_json(indent=2)
    print(output)

To save the result for the MusicDNA-Encoder, write the JSON returned by
``result.to_json()`` to a file such as ``detector-output.json``. For example::

    from pathlib import Path

    Path("detector-output.json").write_text(
        result.to_json(indent=2),
        encoding="utf-8",
    )

The Encoder can read that file directly.

The same operation can be run directly from a console::

    python -c "from musicdna_detector import AnalysisConfig, analyze_file; print(analyze_file('examples/synthetic_melody.wav', AnalysisConfig(profile='humming')).to_json(indent=2))"

The JSON output is the ``musicdna-analysis-v0`` contract and can be passed to
MusicDNA-Encoder. The default decoder uses SoundFile/libsndfile and downmixes
multichannel input to mono.

Additional example audio
------------------------

For a humming recording, see the `HumTrans dataset
<https://huggingface.co/datasets/dadinghh2/HumTrans>`_; for a singing
recording, use the `VocalSet audio release
<https://doi.org/10.5281/zenodo.1442513>`_. The related `Annotated-VocalSet
<https://doi.org/10.5281/zenodo.7061507>`_ provides reference annotations for
those VocalSet files rather than replacing the audio download. Both links point
to datasets rather than individual audio-file downloads. The included synthetic
WAV above is enough to verify an installation without downloading either
dataset.

Known limits
------------

The detector targets short, predominantly monophonic vocal queries. Pitch and
note boundaries depend on recording quality, voicing, background sound, and
the selected profile. The output is a note-event representation, not a full
score: lyrics, meter, accompaniment separation, and musical structure are
outside its scope.

License
-------

The project is distributed under the MIT License. See ``LICENSE`` and
``THIRD_PARTY_NOTICES.txt`` for the software license and DSP provenance.
