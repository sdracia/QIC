def checkpoint(debug, msg1="ERROR", msg2="Checkpoint reached.", stop=False):
  """
  Prints a checkpoint message and optionally raises an error if debugging is enabled.
  
  Parameters
  ----------
  debug : bool
    If True, the checkpoint message will be printed.
  msg1 : str, optional
    The message to display at the checkpoint (title). Default is "ERROR."
  msg2 : str, optional
    Detailed message to display at the checkpoint. Default is "Checkpoint reached".
  stop : bool, optional
    If True, raises ValueError to stop execution. Default is False.
  """
  if debug:
    print(f"Checkpoint: {msg1}")
    if stop:
      raise ValueError(f"Execution stopped at checkpoint: {msg1}\n{msg2}")