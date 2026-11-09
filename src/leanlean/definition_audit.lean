import Lean

/-!
Kernel audit for the LeanLean definition-equivalence stage, run with `lake env lean --run`.

`definition_audit.lean MODE MODULE NAMES_JSON TRUSTED_PREFIX...` imports MODULE without
environment extensions, so nothing from the submission runs here. Modules whose `.olean`
resolves under a TRUSTED_PREFIX (toolchain, pinned packages) are trusted; every other
reachable constant is replayed in a fresh kernel before its result is reported.

* `reference`: classify each `LeanLeanReference.<name>` as a definition or an intended
  hole (its value uses `sorryAx`).
* `check`: verify `LeanLeanDefCheck.«<name>» : @<name> = @LeanLeanReference.<name>`
  for each name, using only the standard axioms.
-/

open Lean

def standardAxioms : List Name := [``propext, ``Classical.choice, ``Quot.sound]

def referenceName (name : Name) : Name := `LeanLeanReference ++ name

def checkName (raw : String) : Name := .str `LeanLeanDefCheck raw

def isTrusted (env : Environment) (trusted : NameSet) (name : Name) : Bool :=
  match env.getModuleIdxFor? name with
  | some idx => trusted.contains env.header.moduleNames[idx.toNat]!
  | none => false

/-- Constants that are not trusted and are reachable from `roots`, plus every axiom referenced. -/
def closure (env : Environment) (trusted : NameSet) (roots : Array Name) :
    Except String (Std.HashMap Name ConstantInfo × NameSet) := Id.run do
  let mut seen : NameSet := {}
  let mut todo := roots
  let mut consts : Std.HashMap Name ConstantInfo := {}
  let mut axioms : NameSet := {}
  while !todo.isEmpty do
    let name := todo.back!
    todo := todo.pop
    if seen.contains name then
      continue
    seen := seen.insert name
    let some info := env.find? name
      | return .error s!"unknown constant {name}"
    if info matches .axiomInfo _ then
      axioms := axioms.insert name
    if isTrusted env trusted name then
      continue
    consts := consts.insert name info
    todo := todo ++ info.getUsedConstantsAsSet.toArray
    -- The kernel adds an inductive together with its constructors and checks its recursors.
    match info with
    | .inductInfo val => todo := todo ++ val.all.toArray ++ val.ctors.toArray |>.push (val.name.str "rec")
    | .ctorInfo val => todo := todo.push val.induct
    | .recInfo val => todo := todo ++ val.all.toArray
    | _ => pure ()
  return .ok (consts, axioms)

def usesSorry (env : Environment) (trusted : NameSet) (name : Name) : Bool :=
  match closure env trusted #[name] with
  | .ok (_, axioms) => axioms.contains ``sorryAx
  | .error _ => true

def distinctParams (levels : List Level) : Bool :=
  levels.all (·.isParam) && levels.eraseDups.length == levels.length

/-- The statement must be exactly `@name.{us} = @LeanLeanReference.name.{us}`. -/
def statementOk (env : Environment) (name : Name) (info : ConstantInfo) : Bool :=
  match info, info.type.eq? with
  | .thmInfo _, some (_, .const post us, .const ref vs) =>
    post == name && ref == referenceName name && us == vs && distinctParams us
      && (env.find? name |>.map (·.levelParams.length)) == some us.length
  | _, _ => false

def reportReference (env : Environment) (trusted : NameSet) (names : Array String) : Json :=
  Json.mkObj <| names.toList.map fun raw =>
    let ref := referenceName raw.toName
    let status := match env.find? ref with
      | some (.defnInfo _) => if usesSorry env trusted ref then "intended_hole" else "definition"
      | some _ => "not_a_definition"
      | none => "missing_reference"
    (raw, Json.mkObj [("status", status)])

def reportCheck (env : Environment) (trusted : NameSet) (names : Array String) : IO Json := do
  let roots := names.map checkName |>.filter (env.contains ·)
  let (consts, _) ← IO.ofExcept (closure env trusted roots)
  -- Replay over the trusted imports only; this rechecks the submission's constants as well.
  let modules := env.header.moduleNames.filter trusted.contains
  let base ← importModules (modules.map ({ module := · })) {} (loadExts := false)
  discard <| base.toKernelEnv.replay consts
  let rows ← names.toList.mapM fun raw => do
    let chk := checkName raw
    let row := match env.find? chk with
      | none => Json.mkObj [("status", "missing_proof")]
      | some info =>
        if !statementOk env raw.toName info then
          Json.mkObj [("status", "bad_statement")]
        else
          match closure env trusted #[chk] with
          | .error e => Json.mkObj [("status", "error"), ("error", e)]
          | .ok (_, axioms) =>
            let axioms := axioms.toList.map toString
            let extra := axioms.filter (fun a => !standardAxioms.any (·.toString == a))
            let status := if extra.contains "sorryAx" then "not_proved"
              else if extra.isEmpty then "equal" else "nonstandard_axioms"
            Json.mkObj [("status", status), ("axioms", toJson axioms)]
    return (raw, row)
  return Json.mkObj [("replayed_constants", consts.size), ("definitions", Json.mkObj rows)]

def main (args : List String) : IO UInt32 := do
  let mode :: module :: namesPath :: prefixes := args
    | throw (IO.userError "usage: MODE MODULE NAMES_JSON TRUSTED_PREFIX...")
  initSearchPath (← findSysroot)
  let names ← IO.ofExcept <| (Json.parse (← IO.FS.readFile namesPath)) >>= fromJson? (α := Array String)
  let env ← importModules #[{ module := module.toName }] {} (loadExts := false)
  let mut trusted : NameSet := {}
  for mod in env.header.moduleNames do
    let path := (← IO.FS.realPath (← findOLean mod)).toString
    if prefixes.any (path.startsWith ·) then
      trusted := trusted.insert mod
  let report ← match mode with
    | "reference" => pure (reportReference env trusted names)
    | "check" => reportCheck env trusted names
    | _ => throw (IO.userError s!"unknown mode {mode}")
  IO.println report.compress
  return 0
