class SkylarkError(Exception):
  pass

class DimensionMistmatchError(SkylarkError):
  pass

class LowerLayerError(SkylarkError):
  pass

class UnsupportedError(SkylarkError):
  pass

class ParameterMistmatchError(SkylarkError):
  pass

class InvalidParamterError(SkylarkError):
  pass

class InvalidObjectError(SkylarkError):
  pass
