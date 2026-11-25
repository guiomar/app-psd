<h1 align="center">PSD of raw data</h1>

<p align="center">
  Computes the power spectral density of raw MEG/EEG data with MNE-Python's Welch method, saving per-channel PSD tables for gradiometers, magnetometers and EEG and PSD plots.
</p>

<p align="center">
  <a href="https://github.com/brainlife/abcd-spec"><img src="https://img.shields.io/badge/ABCD_Spec-v1.1-green.svg" alt="ABCD Spec v1.1"></a>
  <a href="https://doi.org/10.25663/brainlife.app.530"><img src="https://img.shields.io/badge/Brainlife-brainlife.app.530-blue.svg" alt="Run on brainlife.io"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-see_LICENSE-lightgrey.svg" alt="License"></a>
</p>

## At a glance

<!-- brainlife-app:begin (generated from the App manifest; edit the manifest, not this block) -->

> **Computes the power spectral density of raw MEG/EEG data with MNE-Python's Welch method, saving per-channel PSD tables for gradiometers, magnetometers and EEG and PSD plots.**

Reads an MNE raw file and runs mne.time_frequency.psd_welch separately for each channel type present (grad, mag, eeg) between fmin and fmax, with configurable time window, FFT length, overlap, window and averaging. PSDs are stored in dB as TSV tables (channels x frequencies) and plotted with MNE and with a manual plot. Use it to inspect spectral content or as input to peak-frequency Apps.

| | |
|---|---|
| **Inputs** | `neuro/meeg/mne/raw` |
| **Outputs** | `neuro/meeg/psd` ×3 · `raw` |
| **Container** | `brainlife/mne:0.23dev` |
| **Resources** | 4 CPUs · 64 GB memory · 1 h walltime |
| **Run it** | [doi.org/10.25663/brainlife.app.530](https://doi.org/10.25663/brainlife.app.530) |

#### Inputs

| Input | Datatype | Required | Description |
|---|---|---|---|
| `megfif` | `neuro/meeg/mne/raw` | yes | Raw MEG/EEG recording (MNE raw, FIF). |

#### Outputs

| Output | Datatype | Description |
|---|---|---|
| `out_grad` | `neuro/meeg/psd` | PSD table (TSV, channels x frequencies, dB) for gradiometer channels. |
| `out_mag` | `neuro/meeg/psd` | PSD table (TSV, channels x frequencies, dB) for magnetometer channels. |
| `out_eeg` | `neuro/meeg/psd` | PSD table (TSV, channels x frequencies, dB) for EEG channels. |
| `out_figs` | `raw` | PSD figures (PNG) produced with MNE and with a manual plot. |

<details>
<summary><b>Parameters</b> (12)</summary>

| Parameter | Type | Default | Description |
|---|---|---|---|
| `fmin` | number | `0` | Min frequency of interest. |
| `fmax` | number | `100` | Max frequency of interest. |
| `tmin` | number |  | Min time of interest. |
| `tmax` | number |  | Max time of interest. |
| `n_fft` | number | `4096` | The length of FFT used, must be >= n_per_seg. The segments will be zero-padded if n_fft > n_per_seg. If n_per_seg is None, n_fft must be <= number of time points in the data. |
| `n_overlap` | number | `2000` | The number of points of overlap between segments. Will be adjusted to be <= n_per_seg. |
| `n_per_seg` | number |  | Length of each Welch segment (windowed with a Hamming window). Defaults to None, which sets n_per_seg equal to n_fft |
| `window` | string | `hamming` | Windowing function to use. See scipy.signal.get_window(). |
| `proj` | boolean | `false` | Apply SSP projection vectors. |
| `reject_by_annotation` | boolean | `true` | Whether to omit bad segments from the data before fitting. If True (default), annotated segments whose description begins with 'bad' are omitted. If False, no rejection based on annotations is performed. |
| `picks` | string | `None` | Channels to include. In lists, channel type strings (e.g., [meg, eeg]) will pick channels of those types, channel name strings (e.g., [MEG0111, MEG2623]) will pick the given channels. When you use a list, don't forget to use the square brackets. Can also be the string values “all” to pick all channels, or “data” to pick data channels. None will pick all data channels. Note that channels in info['bads'] will be included if their names are explicitly provided. |
| `average` | enum | `mean` | How to average the segments. - If 'mean (default), calculate the arithmetic mean. - If 'median', calculate the median, corrected for its bias relative to the mean. - If 'None', returns the unaggregated segments. Options: `mean`, `median`, `None`. |

</details>

<!-- brainlife-app:end -->

## Running the App

### On brainlife.io

Open the App at [doi.org/10.25663/brainlife.app.530](https://doi.org/10.25663/brainlife.app.530), choose `neuro/meeg/mne/raw` from your project, set the parameters and press **Execute**. The results are saved to your project as `neuro/meeg/psd`, `neuro/meeg/psd`, `neuro/meeg/psd`, `raw`.

### Locally

You need [Singularity/Apptainer](https://apptainer.org/) (the App runs in `brainlife/mne:0.23dev`), [`jq`](https://jqlang.github.io/jq/).

1. Clone this repository.
2. Create `config.json` with the paths to your inputs and the parameters:

   ```json
   {
       "mne": "/path/to/megfif/mne",
       "fmin": 0,
       "fmax": 100,
       "tmin": "<number>",
       "tmax": "<number>",
       "n_fft": 4096,
       "n_overlap": 2000,
       "n_per_seg": "<number>",
       "window": "hamming",
       "proj": false,
       "reject_by_annotation": true,
       "picks": "None",
       "average": "mean"
   }
   ```

3. Run `./main`.

## app-peak-frequency documentation

2) Input file is:
    * `meg/fif` meg data file
5) Ouput files are:
    * A series of files `.tsv` file containing the PSD, 
    for gradiometers (grad), magnetometers (mag) and electroencephalography (eeg)
    acording to the channels types contained in raw meg data
    * a plot of the PSD for all channels (grad, mag, eeg) using MNE-Python function
    * a plot of the PSD for all channels (grad, mag, eeg) manually implemented (for reproducibility purposes)

## Citing

If you use this App, please cite it, the software it builds on and brainlife.io:

- PSD of raw data. brainlife.io App. [doi:10.25663/brainlife.app.530](https://doi.org/10.25663/brainlife.app.530)
- Gramfort A et al. MEG and EEG data analysis with MNE-Python. Frontiers in Neuroscience 7:267 (2013). [doi:10.3389/fnins.2013.00267](https://doi.org/10.3389/fnins.2013.00267)
- Hayashi S, Caron BA, Heinsfeld AS, et al. brainlife.io: a decentralized and open-source cloud platform to support neuroscience research. *Nature Methods* 21:809–813 (2024). [doi:10.1038/s41592-024-02237-2](https://doi.org/10.1038/s41592-024-02237-2)
- Avesani P, McPherson B, Hayashi S, et al. The open diffusion data derivatives, brain data upcycling via integrated publishing of derivatives and reproducible open cloud services. *Scientific Data* 6:69 (2019). [doi:10.1038/s41597-019-0073-y](https://doi.org/10.1038/s41597-019-0073-y)

## Authors

- Guiomar Niso

## Funding

[![Wellcome-226486/Z/22/Z](https://img.shields.io/badge/Wellcome-226486%2FZ%2F22%2FZ-purple.svg)](https://wellcome.org/)
[![NIH-NINDS-U24NS140384](https://img.shields.io/badge/NIH_NINDS-U24NS140384-green.svg)](https://reporter.nih.gov/project-details/11261037)
[![NIH-NINDS-UM1NS132207](https://img.shields.io/badge/NIH_NINDS-UM1NS132207-green.svg)](https://reporter.nih.gov/project-details/11411328)
[![NIH-NIBIB-R01EB029272](https://img.shields.io/badge/NIH_NIBIB-R01EB029272-green.svg)](https://reporter.nih.gov/project-details/10262925)
[![NIH-NIBIB-R01EB030896](https://img.shields.io/badge/NIH_NIBIB-R01EB030896-green.svg)](https://reporter.nih.gov/project-details/10428625)
[![NIH-NIMH-R01MH126699](https://img.shields.io/badge/NIH_NIMH-R01MH126699-green.svg)](https://reporter.nih.gov/project-details/10460628)
[![NIH-NIDCD-R01DC020088](https://img.shields.io/badge/NIH_NIDCD-R01DC020088-green.svg)](https://reporter.nih.gov/project-details/11140284)
[![NSF-BCS-1734853](https://img.shields.io/badge/NSF_BCS-1734853-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1734853)
[![NSF-BCS-2203524](https://img.shields.io/badge/NSF_BCS-2203524-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2203524)
[![NSF-IIS-1636893](https://img.shields.io/badge/NSF_IIS-1636893-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1636893)
[![NSF-OAC-2148729](https://img.shields.io/badge/NSF_OAC-2148729-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2148729)
[![NSF-IIS-1912270](https://img.shields.io/badge/NSF_IIS-1912270-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1912270)
[![NSF-IIS-2148700](https://img.shields.io/badge/NSF_IIS-2148700-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2148700)
[![NSF-OAC-1916518](https://img.shields.io/badge/NSF_OAC-1916518-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1916518)

<sub>brainlife.io, which runs this App, is publicly funded by Wellcome, the U.S. National Institutes of Health and the U.S. National Science Foundation; acknowledging that support helps keep it running.</sub>

## License

See [LICENSE](LICENSE).
