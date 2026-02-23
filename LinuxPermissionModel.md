# The Linux Permission Model

The Linux permission model is the **"bouncer"** of the operating system. It determines exactly who can look at, change, or run a file. Every single file and directory in Linux has an owner and a set of rules attached to it.

When you run `ls -l`, you see these rules in a string that looks like this: `-rwxr-xr--`.

---

## 1. The Three "Who's" (Classes)
Permissions are assigned to three distinct categories of users:

* **User (u):** The individual person who owns the file (usually the creator).
* **Group (g):** A collection of users who share the same access (e.g., the "developers" group).
* **Others (o):** Everyone else on the system ("the world").

---

## 2. The Three "What's" (Capabilities)
There are three basic types of access you can grant:

| Permission | Symbol | Number | Effect on Files | Effect on Directories |
| :--- | :--- | :--- | :--- | :--- |
| **Read** | `r` | 4 | Can view the file's contents. | Can list the files inside (`ls`). |
| **Write** | `w` | 2 | Can modify or delete the file. | Can add, delete, or rename files inside. |
| **Execute** | `x` | 1 | Can run the file as a program/script. | Can "enter" the directory (`cd`). |



---

## 3. Reading the "Permission String"
If you see `-rwxr-xr--`, break it down into four parts:

* **Type:** The first character (`-` for file, `d` for directory).
* **Owner:** The next three (`rwx`) — the owner can do everything.
* **Group:** The middle three (`r-x`) — the group can read and execute, but not change it.
* **Others:** The last three (`r--`) — everyone else can only read it.



---

## 4. The Numeric (Octal) Method
System administrators often use numbers instead of letters because they are faster to type. You simply add the numbers of the permissions you want to grant:

* **7** ($4+2+1$): Full permissions (`rwx`)
* **6** ($4+2$): Read and Write (`rw-`)
* **5** ($4+1$): Read and Execute (`r-x`)
* **4**: Read-only (`r--`)

**Example:** To give the owner full access (7), the group read access (4), and others nothing (0), you would use the command:  
`chmod 740 filename`

---

## 5. Changing Permissions
You use two main commands to manage this:

* **`chmod` (Change Mode):** Changes the permissions (the `rwx`).
* **`chown` (Change Owner):** Changes who the owner or the group is.