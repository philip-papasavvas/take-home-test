# Running the script
Steps for running the python code.

- Download the codebase/git clone the codebase
- Run Docker desktop on the machine you will run the code on
- Open the terminal in the take-home-test folder within the code base, containing the requirements.txt and task_submission.py files.
- Run the following command in the terminal: `docker build -t take-home-test .` 
*(This will take a few minutes to download the requirements and build the Docker image).*
- Then run the following command to run the code: 
- `docker run --name take_home_test_container \
  -v "$(pwd)"/data:/app/data \
  -v "$(pwd)"/output:/app/output \
  take-home-test`
- Once this has been run on the container, you should see that two files, `data_analysis.log` and `output_file.csv` have been populated in the output
folder